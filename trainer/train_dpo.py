"""
QwenSearch DPO 偏好对齐训练
在 SFT 模型基础上，使用启发式引导 vs 直接给答案的偏好对进行 DPO 训练
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler

from model.qwen_vlm import QwenSearchVLM, QwenSearchConfig
from dataset.edu_dataset import EduDPODataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, init_distributed_mode, setup_seed,
    get_model_params, SkipBatchSampler, edu_dpo_collate_fn,
)

warnings.filterwarnings('ignore')


def dpo_loss(
    chosen_log_probs, chosen_seq_lens,
    rejected_log_probs, rejected_seq_lens,
    ref_chosen_log_probs, ref_rejected_log_probs,
    beta=0.1, label_smoothing=0.0,
):
    """
    DPO Loss 计算

    Args:
        chosen_log_probs: [B] 当前模型 chosen 的 log-prob
        chosen_seq_lens: [B] chosen 序列长度
        rejected_log_probs: [B] 当前模型 rejected 的 log-prob
        rejected_seq_lens: [B] rejected 序列长度
        ref_chosen_log_probs: [B] 参考模型 chosen 的 log-prob
        ref_rejected_log_probs: [B] 参考模型 rejected 的 log-prob
        beta: KL 惩罚系数
        label_smoothing: 标签平滑

    Returns:
        loss, chosen_rewards, rejected_rewards, accuracy
    """
    chosen_log_probs = chosen_log_probs / chosen_seq_lens.clamp(min=1)
    rejected_log_probs = rejected_log_probs / rejected_seq_lens.clamp(min=1)
    ref_chosen_log_probs = ref_chosen_log_probs / chosen_seq_lens.clamp(min=1)
    ref_rejected_log_probs = ref_rejected_log_probs / rejected_seq_lens.clamp(min=1)

    chosen_rewards = beta * (chosen_log_probs - ref_chosen_log_probs)
    rejected_rewards = beta * (rejected_log_probs - ref_rejected_log_probs)
    logits = chosen_rewards - rejected_rewards

    loss = -torch.nn.functional.logsigmoid(logits).mean()
    accuracy = (logits > 0).float().mean()

    if label_smoothing > 0:
        loss = (1 - label_smoothing) * loss + label_smoothing * (
            -torch.nn.functional.logsigmoid(-logits).mean()
        )

    return loss, chosen_rewards.mean(), rejected_rewards.mean(), accuracy


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step

    for step, batch_data in enumerate(loader, start=start_step + 1):
        (chosen_ids, chosen_mask, chosen_labels,
         rejected_ids, rejected_mask, rejected_labels,
         pixel_values, image_grid_thw) = batch_data

        chosen_ids = chosen_ids.to(args.device)
        chosen_mask = chosen_mask.to(args.device)
        chosen_labels = chosen_labels.to(args.device)
        rejected_ids = rejected_ids.to(args.device)
        rejected_mask = rejected_mask.to(args.device)
        rejected_labels = rejected_labels.to(args.device)
        pixel_values = pixel_values.to(args.device)
        image_grid_thw = image_grid_thw.to(args.device)
        last_step = step

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 1. 当前模型计算 log-prob
        with autocast_ctx:
            chosen_log_probs, chosen_seq_lens, _ = model.get_log_probs(
                chosen_ids, chosen_labels, pixel_values, image_grid_thw
            )
            rejected_log_probs, rejected_seq_lens, _ = model.get_log_probs(
                rejected_ids, rejected_labels, pixel_values, image_grid_thw
            )

            # 2. 参考模型计算 log-prob
            with torch.no_grad():
                ref_chosen_log_probs, _, _ = ref_model.get_log_probs(
                    chosen_ids, chosen_labels, pixel_values, image_grid_thw
                )
                ref_rejected_log_probs, _, _ = ref_model.get_log_probs(
                    rejected_ids, rejected_labels, pixel_values, image_grid_thw
                )

            # 3. DPO loss
            loss, chosen_reward, rejected_reward, acc = dpo_loss(
                chosen_log_probs, chosen_seq_lens,
                rejected_log_probs, rejected_seq_lens,
                ref_chosen_log_probs, ref_rejected_log_probs,
                beta=args.beta, label_smoothing=args.label_smoothing,
            )
            loss = loss / args.accumulation_steps

        loss.backward()

        if step % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'[DPO] Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'loss:{loss.item() * args.accumulation_steps:.4f}, '
                   f'c_reward:{chosen_reward.item():.4f}, r_reward:{rejected_reward.item():.4f}, '
                   f'acc:{acc.item():.4f}, lr:{current_lr:.8f}, eta:{eta_min:.1f}min')
            if wandb:
                wandb.log({
                    "dpo_loss": loss.item() * args.accumulation_steps,
                    "chosen_reward": chosen_reward.item(),
                    "rejected_reward": rejected_reward.item(),
                    "accuracy": acc.item(),
                    "learning_rate": current_lr,
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            ckp = os.path.join(args.save_dir, args.save_weight)
            os.makedirs(ckp, exist_ok=True)
            model.save_pretrained(ckp)
            Logger(f'[DPO] Model saved to {ckp}')

        del batch_data, loss, chosen_reward, rejected_reward, acc

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenSearch DPO Training")
    parser.add_argument("--model_name", type=str, default="./model/Qwen2-VL-2B-Instruct", help="基座模型路径")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", type=str, default="edu_dpo", help="保存权重的名称")
    parser.add_argument("--from_weight", type=str, default="../out/edu_sft", help="SFT 权重路径（用于初始化和参考模型）")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="学习率")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL 惩罚系数")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="标签平滑")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=200, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--data_path", type=str, default="../dataset/edu_science.parquet", help="训练数据路径")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb/swanlab")
    parser.add_argument("--wandb_project", type=str, default="QwenSearch-DPO", help="wandb 项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)

    # ========== 2. 混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 3. wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb
        wandb_run_name = f"QwenSearch-DPO-E{args.epochs}-B{args.batch_size}-Beta{args.beta}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    # ========== 4. 加载模型 -- 从 SFT checkpoint 加载 ==========
    Logger(f'[DPO] Loading base model from: {args.model_name}')
    Logger(f'[DPO] Loading SFT weights from: {args.from_weight}')
    config = QwenSearchConfig(
        model_name_or_path=args.model_name,
        use_lora=True,
        max_seq_len=args.max_seq_len,
    )
    model = QwenSearchVLM(config)
    # 加载 SFT 权重
    if os.path.exists(args.from_weight):
        from peft import PeftModel
        model.base_model = PeftModel.from_pretrained(
            model.base_model, args.from_weight
        )
    model = model.to(args.device)

    # 参考模型：加载相同 SFT 权重，完全冻结
    ref_model = QwenSearchVLM(config)
    if os.path.exists(args.from_weight):
        from peft import PeftModel
        ref_model.base_model = PeftModel.from_pretrained(
            ref_model.base_model, args.from_weight
        )
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model = ref_model.to(args.device)
    ref_model.eval()

    get_model_params(model)
    Logger(f'[DPO] beta={args.beta}, lr={args.learning_rate}')

    # ========== 5. 加载数据 ==========
    train_ds = EduDPODataset(
        parquet_path=args.data_path,
        processor=model.processor,
        max_length=args.max_seq_len,
    )
    Logger(f'[DPO] Dataset: {len(train_ds)} samples')
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # ========== 6. 优化器 ==========
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    # ========== 7. 开始训练 ==========
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, 0)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=edu_dpo_collate_fn,
        )
        train_epoch(epoch, loader, len(loader), 0, wandb)

    if is_main_process():
        ckp = os.path.join(args.save_dir, args.save_weight)
        os.makedirs(ckp, exist_ok=True)
        model.save_pretrained(ckp)
        Logger(f'[DPO] Final model saved to {ckp}')

    if dist.is_initialized():
        dist.destroy_process_group()