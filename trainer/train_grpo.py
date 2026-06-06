"""
QwenSearch GRPO 强化优化训练
使用 LLM-as-Judge API 作为奖励函数，通过组内相对优势优化模型策略
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
from dataset.edu_dataset import EduGRPODataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, init_distributed_mode, setup_seed,
    get_model_params, SkipBatchSampler, edu_grpo_collate_fn,
)
from trainer.llm_reward import APILLMRewardModel
from trainer.reward_model import edu_grpo_advantage, edu_grpo_policy_loss

warnings.filterwarnings('ignore')


@torch.no_grad()
def generate_responses(
    model, prompt_ids, prompt_mask, pixel_values, image_grid_thw,
    processor, num_generations=4, max_new_tokens=256,
):
    """
    对单个 prompt 生成 K 个候选回答
    """
    model.eval()
    responses = []

    for i in range(num_generations):
        temperature = 0.7 + 0.1 * i
        gen_ids = model.generate(
            input_ids=prompt_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            do_sample=True,
        )
        prompt_len = prompt_ids.shape[1]
        new_tokens = gen_ids[0, prompt_len:]
        response_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

        del gen_ids

        if len(response_text.strip()) > 0:
            responses.append(response_text)

    return responses


def train_epoch(epoch, loader, iters, args, model, optimizer, reward_model,
                 start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step

    for step, (prompt_ids, prompt_mask, gt_ids, pixel_values, image_grid_thw) in enumerate(loader, start=start_step + 1):
        prompt_ids = prompt_ids.to(args.device)
        prompt_mask = prompt_mask.to(args.device)
        gt_ids = gt_ids.to(args.device)
        pixel_values = pixel_values.to(args.device)
        image_grid_thw = image_grid_thw.to(args.device)
        last_step = step

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        batch_size = prompt_ids.size(0)
        total_loss = 0.0
        all_rewards = []

        for b in range(batch_size):
            single_prompt = prompt_ids[b:b + 1]
            single_pv = pixel_values[b:b + 1]
            single_thw = image_grid_thw[b:b + 1]
            single_gt = gt_ids[b]

            # 解码 GT 文本
            gt_text = model.processor.tokenizer.decode(
                single_gt[single_gt != model.processor.tokenizer.pad_token_id],
                skip_special_tokens=True,
            )

            # 1. 生成 K 个候选回答
            response_texts = generate_responses(
                model,
                single_prompt, prompt_mask[b:b + 1],
                single_pv, single_thw,
                model.processor,
                num_generations=args.num_generations,
                max_new_tokens=args.max_new_tokens,
            )

            if len(response_texts) == 0:
                continue

            # 2. 计算奖励
            gt_texts = [gt_text] * len(response_texts)
            rewards = reward_model.compute_group_rewards(response_texts, gt_texts)
            rewards = rewards.to(args.device)
            advantages = edu_grpo_advantage(rewards)
            all_rewards.extend(rewards.tolist())

            # 3. 批量 tokenize（避免重复计算）
            resp_ids_list = []
            for resp_text in response_texts:
                resp_ids = model.processor.tokenizer(
                    resp_text, add_special_tokens=False, return_tensors="pt"
                ).input_ids[0]
                full_ids = torch.cat([single_prompt[0], resp_ids])
                if len(full_ids) > args.max_seq_len:
                    full_ids = full_ids[:args.max_seq_len]
                resp_ids_list.append(full_ids)

            if len(resp_ids_list) == 0:
                continue

            # 4. 批量计算旧策略 log probs
            old_log_probs_list = []
            with torch.no_grad():
                for full_ids in resp_ids_list:
                    full_ids = full_ids.unsqueeze(0)
                    labels = full_ids.clone()
                    labels[0, :single_prompt.shape[1]] = -100
                    log_probs, seq_len, _ = model.get_log_probs(
                        full_ids, labels, single_pv, single_thw
                    )
                    old_log_probs_list.append(log_probs.squeeze(0) / seq_len.clamp(min=1))

            if len(old_log_probs_list) == 0:
                continue

            # 5. 批量计算新策略 log probs
            new_log_probs_list = []
            for full_ids in resp_ids_list:
                full_ids = full_ids.unsqueeze(0)
                labels = full_ids.clone()
                labels[0, :single_prompt.shape[1]] = -100
                log_probs, seq_len, _ = model.get_log_probs(
                    full_ids, labels, single_pv, single_thw
                )
                new_log_probs_list.append(log_probs.squeeze(0) / seq_len.clamp(min=1))

            if len(new_log_probs_list) == 0:
                continue

            old = torch.stack(old_log_probs_list)
            new = torch.stack(new_log_probs_list)

            # 6. GRPO policy loss
            group_loss = edu_grpo_policy_loss(old, new, advantages[:len(old)], args.clip_eps)
            total_loss += group_loss

        if batch_size > 0:
            total_loss = total_loss / batch_size
        total_loss = total_loss / args.accumulation_steps

        total_loss.backward()

        if step % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_lr = optimizer.param_groups[-1]['lr']
            avg_reward = sum(all_rewards) / max(len(all_rewards), 1) if all_rewards else 0.0
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'[GRPO] Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'loss:{total_loss.item() * args.accumulation_steps:.4f}, '
                   f'avg_reward:{avg_reward:.4f}, lr:{current_lr:.8f}, eta:{eta_min:.1f}min')
            if wandb:
                wandb.log({
                    "grpo_loss": total_loss.item() * args.accumulation_steps,
                    "avg_reward": avg_reward,
                    "learning_rate": current_lr,
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            ckp = os.path.join(args.save_dir, args.save_weight)
            os.makedirs(ckp, exist_ok=True)
            model.save_pretrained(ckp)
            Logger(f'[GRPO] Model saved to {ckp}')

        del prompt_ids, prompt_mask, gt_ids, pixel_values, image_grid_thw, total_loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenSearch GRPO Training")
    parser.add_argument("--model_name", type=str, default="./model/Qwen2-VL-2B-Instruct", help="基座模型路径")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", type=str, default="edu_grpo", help="保存权重的名称")
    parser.add_argument("--from_weight", type=str, default="../out/edu_sft", help="SFT 权重路径")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="学习率")
    parser.add_argument("--num_generations", type=int, default=4, help="每个 prompt 生成候选数 K")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="生成的最大 token 数")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip 范围")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=5, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--data_path", type=str, default="../dataset/edu_science.parquet", help="训练数据路径")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb/swanlab")
    parser.add_argument("--wandb_project", type=str, default="QwenSearch-GRPO", help="wandb 项目名")
    parser.add_argument("--api_key", type=str, default="", help="LLM API Key（默认读取 OPENAI_API_KEY 环境变量）")
    parser.add_argument("--api_model", type=str, default="gpt-4o-mini", help="LLM 奖励模型名称")
    parser.add_argument("--api_base_url", type=str, default=None, help="LLM API 地址（兼容 OpenAI 格式）")
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
        wandb_run_name = f"QwenSearch-GRPO-E{args.epochs}-K{args.num_generations}-LR{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    # ========== 4. 加载模型 ==========
    Logger(f'[GRPO] Loading base model from: {args.model_name}')
    Logger(f'[GRPO] Loading checkpoint from: {args.from_weight}')
    config = QwenSearchConfig(
        model_name_or_path=args.model_name,
        use_lora=True,
        max_seq_len=args.max_seq_len,
    )
    model = QwenSearchVLM(config)
    if os.path.exists(args.from_weight):
        from peft import PeftModel
        model.base_model = PeftModel.from_pretrained(
            model.base_model, args.from_weight
        )
    model = model.to(args.device)
    model.train()

    # 奖励模型: LLM-as-Judge API
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    reward_model = APILLMRewardModel(
        api_key=api_key,
        model=args.api_model,
        base_url=args.api_base_url,
    )
    Logger(f'[GRPO] Reward model: API ({args.api_model})')

    get_model_params(model)
    Logger(f'[GRPO] K={args.num_generations}, clip_eps={args.clip_eps}, lr={args.learning_rate}')

    # ========== 5. 加载数据 ==========
    train_ds = EduGRPODataset(
        parquet_path=args.data_path,
        processor=model.processor,
        max_length=args.max_seq_len,
    )
    Logger(f'[GRPO] Dataset: {len(train_ds)} samples')
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
            collate_fn=edu_grpo_collate_fn,
        )
        train_epoch(epoch, loader, len(loader), args, model, optimizer, reward_model,
                    0, wandb)

    if is_main_process():
        ckp = os.path.join(args.save_dir, args.save_weight)
        os.makedirs(ckp, exist_ok=True)
        model.save_pretrained(ckp)
        Logger(f'[GRPO] Final model saved to {ckp}')

    if dist.is_initialized():
        dist.destroy_process_group()