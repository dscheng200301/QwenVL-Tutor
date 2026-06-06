"""
QwenVL-Tutor SFT 微调训练
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

from model.qwen_vlm import QwenVLTutor, QwenVLTutorConfig
from dataset.edu_dataset import EduDataset
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, init_distributed_mode, setup_seed,
    get_model_params, SkipBatchSampler, edu_sft_collate_fn,
)

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, args, model, optimizer, scaler, autocast_ctx,
                 start_step=0, wandb=None, ds_engine=None, fsdp_model=None):
    """
    训练一个 epoch

    Args:
        args: 训练参数
        model: 模型
        optimizer: 优化器
        scaler: 混合精度 GradScaler
        autocast_ctx: 混合精度上下文
        ds_engine: DeepSpeed engine（如果 use_deepspeed=1）
        fsdp_model: FSDP 包装后的模型（如果 use_fsdp=1）
    """
    start_time = time.time()
    last_step = start_step
    use_ds = ds_engine is not None
    use_fsdp = fsdp_model is not None
    # DeepSpeed / FSDP 自己处理混合精度和梯度累积
    use_scaler = scaler is not None and not use_ds

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

        # DeepSpeed 用 bfloat16/fp16 上下文，HF 用 autocast
        if use_ds:
            outputs = ds_engine(
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
            ds_engine.backward(loss)
        else:
            with autocast_ctx:
                if use_fsdp:
                    outputs = fsdp_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        labels=labels,
                    )
                else:
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

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if step % args.accumulation_steps == 0:
            if use_ds:
                ds_engine.step()  # DeepSpeed 自带梯度裁剪
            else:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters() if not use_fsdp else fsdp_model.parameters(),
                    args.grad_clip,
                )
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr'] if not use_ds else ds_engine.get_lr()[0]
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
            if use_ds:
                # DeepSpeed 保存（仅主进程）
                ds_engine.save_checkpoint(ckp)
            else:
                model.save_pretrained(ckp)
            Logger(f'[SFT] Model saved to {ckp}')

        del input_ids, attention_mask, labels, pixel_values, image_grid_thw, outputs, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        if not use_ds:
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters() if not use_fsdp else fsdp_model.parameters(),
                args.grad_clip,
            )
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # ========== 🆕 自动分布式检测 ==========
    # 多卡 + 未启动分布式 + 用户未禁用 → 自动 fork 到 torchrun
    rank = int(os.environ.get("RANK", -1))
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    no_auto = "--no_auto_distributed" in sys.argv
    use_ds = "--use_deepspeed" in sys.argv
    use_fsdp = "--use_fsdp" in sys.argv

    if n_gpus > 1 and rank == -1 and not no_auto and not use_ds and not use_fsdp:
        import subprocess
        master_port = str(29500 + int(__import__('time').time()) % 1000)
        cmd = [
            "torchrun",
            f"--nproc_per_node={n_gpus}",
            f"--master_port={master_port}",
            os.path.abspath(__file__),
        ] + sys.argv[1:]
        print(f"\n{'=' * 70}")
        print(f"[AUTO-DDP] 检测到 {n_gpus} 张 GPU，自动启用 DDP")
        print(f"[AUTO-DDP] 重启到: torchrun --nproc_per_node={n_gpus} --master_port={master_port}")
        print(f"{'=' * 70}\n")
        sys.exit(subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).returncode)

    # 显式添加 no_auto_distributed 参数（用于接收 fork 后的额外参数）
    parser = argparse.ArgumentParser(description="QwenVL-Tutor SFT Training")
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
    parser.add_argument("--wandb_project", type=str, default="QwenVL-Tutor-SFT", help="wandb 项目名")
    # === 分布式训练参数 ===
    parser.add_argument("--use_deepspeed", type=int, default=0, choices=[0, 1], help="是否使用 DeepSpeed (ZeRO)")
    parser.add_argument("--deepspeed_zero_stage", type=int, default=2, choices=[1, 2, 3], help="DeepSpeed ZeRO 阶段")
    parser.add_argument("--deepspeed_offload", type=int, default=0, choices=[0, 1], help="是否启用 CPU offload (ZeRO-2/3)")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="DeepSpeed 配置文件路径（None 时自动生成）")
    parser.add_argument("--use_fsdp", type=int, default=0, choices=[0, 1], help="是否使用 FSDP (PyTorch 内置)")
    parser.add_argument("--gradient_checkpointing", type=int, default=0, choices=[0, 1], help="是否启用梯度检查点（节省显存）")
    parser.add_argument("--no_auto_distributed", action="store_true",
                        help="关闭自动 DDP 检测（多卡时也不自动 fork 到 torchrun）")
    parser.add_argument("--resume", action="store_true",
                        help="从 checkpoint 断点续训练（自动检测 out/{save_weight}/ 中的最新 checkpoint）")
    args = parser.parse_args()

    # ========== 1. 初始化环境 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 设置输出目录 ==========
    os.makedirs(args.save_dir, exist_ok=True)

    # ========== 2.5 断点续训练（resume） ==========
    resume_step = 0
    resume_ckp_path = None
    if args.resume and is_main_process():
        ckp_dir = os.path.join(args.save_dir, args.save_weight)
        import glob
        # 查找所有 checkpoint 子目录
        step_dirs = glob.glob(os.path.join(ckp_dir, "checkpoint-*"))
        if step_dirs:
            # 找到最大的 step
            max_step = 0
            for d in step_dirs:
                try:
                    step_num = int(os.path.basename(d).split("-")[-1])
                    if step_num > max_step:
                        max_step = step_num
                        resume_ckp_path = d
                except:
                    pass
            if resume_ckp_path:
                resume_step = max_step
                Logger(f'[SFT] Resume from checkpoint: {resume_ckp_path} (step {resume_step})')
                # 更新 wandb run name
                if args.use_wandb:
                    wandb_run_name = f"QwenVL-Tutor-SFT-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}-resume{resume_step}"
                # 直接用 checkpoint 路径作为 model_name，后续加载时会用到
                args.model_name = resume_ckp_path
        else:
            Logger(f'[SFT] Resume enabled but no checkpoint found in {ckp_dir}, starting from scratch')

    # 广播 resume_step 到所有进程
    if dist.is_initialized():
        resume_step_tensor = torch.tensor(resume_step, dtype=torch.long, device='cuda')
        dist.broadcast(resume_step_tensor, src=0)
        resume_step = resume_step_tensor.item()

    # ========== 3. 混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import wandb
        wandb_run_name = f"QwenVL-Tutor-SFT-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    # ========== 5. 加载模型 ==========
    Logger(f'[SFT] Loading model from: {args.model_name}')
    config = QwenVLTutorConfig(
        model_name_or_path=args.model_name,
        use_lora=bool(args.use_lora),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_seq_len=args.max_seq_len,
    )
    model = QwenVLTutor(config)
    model = model.to(args.device)
    get_model_params(model)

    # ========== 5.5 分布式包装：DeepSpeed / FSDP ==========
    ds_engine = None
    fsdp_model = None
    optimizer = None  # DeepSpeed 模式下会在后面创建

    if args.use_deepspeed:
        Logger(f'[SFT] 🆕 启用 DeepSpeed ZeRO-{args.deepspeed_zero_stage}')
        from trainer.trainer_utils import generate_deepspeed_config, wrap_model_with_deepspeed
        ds_config_path = args.deepspeed_config or generate_deepspeed_config(
            zero_stage=args.deepspeed_zero_stage,
            offload=bool(args.deepspeed_offload),
            bf16=(args.dtype == 'bfloat16'),
        )
        # 先创建 optimizer
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
        )
        ds_engine, optimizer, _ = wrap_model_with_deepspeed(
            model=model,
            optimizer=optimizer,
            deepspeed_config_path=ds_config_path,
        )
        Logger(f'[SFT] DeepSpeed engine 已创建（ZeRO-{args.deepspeed_zero_stage}）')
    elif args.use_fsdp:
        Logger(f'[SFT] 🆕 启用 FSDP（PyTorch 内置）')
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # FSDP 自动包装策略（按 transformer 层分片）
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16,
            reduce_dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16,
            buffer_dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16,
        )
        fsdp_model = FSDP(
            model,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
        )
        if args.gradient_checkpointing:
            fsdp_model.gradient_checkpointing_enable()
            Logger(f'[SFT] FSDP 梯度检查点已启用')
        Logger(f'[SFT] FSDP 模型包装完成')

    # ========== 6. 加载数据 ==========
    data_paths = args.data_paths.split(",")
    Logger(f'[SFT] Loading datasets: {data_paths}')
    all_datasets = []
    dataset_sizes = []
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
            dataset_sizes.append(len(ds))
            Logger(f'  - {path}: {len(ds)} samples')

    if len(all_datasets) == 1:
        train_ds = all_datasets[0]
    else:
        from torch.utils.data import ConcatDataset
        train_ds = ConcatDataset(all_datasets)

        # 加权采样：平方根倒数加权，使小数据集有更大的采样概率
        # 避免大数据集主导训练
        import math
        dataset_weights = [1.0 / math.sqrt(size) for size in dataset_sizes]
        total_weight = sum(dataset_weights)
        sample_weights = []
        for i, ds in enumerate(all_datasets):
            weight = dataset_weights[i] / total_weight
            sample_weights.extend([weight] * len(ds))

        from torch.utils.data import WeightedRandomSampler
        weighted_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        Logger(f'[SFT] Weighted sampling: weights={[f"{w:.4f}" for w in dataset_weights]}')

    train_sampler = weighted_sampler if len(all_datasets) > 1 else (DistributedSampler(train_ds) if dist.is_initialized() else None)

    # ========== 7. 优化器 ==========
    # DeepSpeed 模式下 optimizer 已在 5.5 步创建
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16' and ds_engine is None))
    if optimizer is None:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.learning_rate,
        )

    # ========== 8. 开始训练 ==========
    # 计算每个 epoch 的 batch 数量，用于断点续训练的全局 step 定位
    batches_per_epoch = len(train_ds) // args.batch_size
    global_resume_step = resume_step  # 累积的全局 step（跨 epoch）

    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()

        # 断点续训练：跳过已完成的全 epoch
        if global_resume_step >= batches_per_epoch:
            global_resume_step -= batches_per_epoch
            continue

        # 断点续训练：计算当前 epoch 需要跳过的 batch 数
        skip_batches = global_resume_step
        global_resume_step = 0  # 只在第一个 epoch 需要跳过部分 batch

        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip_batches)
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=edu_sft_collate_fn,
        )
        train_epoch(epoch, loader, batches_per_epoch, args, model, optimizer, scaler, autocast_ctx,
                    skip_batches, wandb, ds_engine=ds_engine, fsdp_model=fsdp_model)

    # ========== 9. 最终保存 ==========
    if is_main_process():
        ckp = os.path.join(args.save_dir, args.save_weight)
        os.makedirs(ckp, exist_ok=True)
        model.save_pretrained(ckp)
        Logger(f'[SFT] Final model saved to {ckp}')

    if dist.is_initialized():
        dist.destroy_process_group()