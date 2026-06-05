"""
QwenSearch SFT 微调训练
基于 Qwen2-VL 基座，在教育数据集上进行监督微调
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
from dataset.edu_dataset import EduDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, init_distributed_mode, setup_seed,
    get_model_params, SkipBatchSampler, edu_sft_collate_fn,
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step

    for step, (input_ids, attention_mask, labels, pixel_values, image_grid_thw) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        labels = labels.to(args.device)
        pixel_values = pixel_values.to(args.device)
        image_grid_thw = image_grid_thw.to(args.device)
        last_step = step

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
            )
            loss = outputs.loss
            if loss is None:
                continue
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'[SFT] Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'loss:{current_loss:.4f}, lr:{current_lr:.8f}, eta:{eta_min:.1f}min')
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "learning_rate": current_lr,
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            ckp = os.path.join(args.save_dir, f'{args.save_weight}')
            os.makedirs(ckp, exist_ok=True)
            model.save_pretrained(ckp)
            Logger(f'[SFT] Model saved to {ckp}')

        del input_ids, attention_mask, labels, pixel_values, image_grid_thw, outputs, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenSearch SFT Training")
    parser.add_argument("--model_name", type=str, default="./model/Qwen2-VL-2B-Instruct", help="基座模型路径")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument("--save_weight", type=str, default="edu_sft", help="保存权重的名称")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=20, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--data_paths", type=str, 
        default="../dataset/edu_science.parquet,../dataset/edu_math_verse.parquet,../dataset/edu_math_vista.parquet,../dataset/edu_ocr.parquet,../dataset/edu_ceval.parquet,../dataset/edu_cmmlu.parquet,../dataset/edu_ape210k.parquet,../dataset/edu_chartqa.parquet,../dataset/edu_race.parquet,../dataset/edu_openr1_math.parquet,../dataset/edu_gaokao_mathqa.parquet,../dataset/edu_gaokao_mathcloze.parquet", 
        help="数据路径，逗号分隔（默认12个数据集）")
    parser.add_argument("--use_lora", type=int, default=1, choices=[0, 1], help="是否使用 LoRA")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb/swanlab")
    parser.add_argument("--wandb_project", type=str, default="QwenSearch-SFT", help="wandb 项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 设置输出目录 ==========
    os.makedirs(args.save_dir, exist_ok=True)

    # ========== 3. 混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb
        wandb_run_name = f"QwenSearch-SFT-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    # ========== 5. 加载模型 ==========
    Logger(f'[SFT] Loading model from: {args.model_name}')
    config = QwenSearchConfig(
        model_name_or_path=args.model_name,
        use_lora=bool(args.use_lora),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_len=args.max_seq_len,
    )
    model = QwenSearchVLM(config)
    model = model.to(args.device)
    get_model_params(model)

    # ========== 6. 加载数据 ==========
    data_paths = args.data_paths.split(",")
    Logger(f'[SFT] Loading datasets: {data_paths}')
    all_datasets = []
    for path in data_paths:
        path = path.strip()
        if os.path.exists(path):
            ds = EduDataset(
                parquet_path=path,
                processor=model.processor,
                max_length=args.max_seq_len,
                add_system_ratio=1.0,
            )
            all_datasets.append(ds)
            Logger(f'  - {path}: {len(ds)} samples')

    if len(all_datasets) == 1:
        train_ds = all_datasets[0]
    else:
        from torch.utils.data import ConcatDataset
        train_ds = ConcatDataset(all_datasets)

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # ========== 7. 优化器 ==========
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )

    # ========== 8. 开始训练 ==========
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
            collate_fn=edu_sft_collate_fn,
        )
        train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 最终保存 ==========
    if is_main_process():
        ckp = os.path.join(args.save_dir, args.save_weight)
        os.makedirs(ckp, exist_ok=True)
        model.save_pretrained(ckp)
        Logger(f'[SFT] Final model saved to {ckp}')

    if dist.is_initialized():
        dist.destroy_process_group()