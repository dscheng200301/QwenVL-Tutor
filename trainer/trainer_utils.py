"""
训练工具函数集合
"""
import os
import sys
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def is_main_process():
    """判断是否为主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """分布式安全日志输出"""
    if is_main_process():
        print(content)


def get_lr(current_step: int, total_steps: int, lr: float):
    """Cosine 学习率调度"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / max(total_steps, 1))))


def init_distributed_mode():
    """
    初始化分布式训练模式（支持 torchrun / accelerate launch / 单卡）
    返回 local_rank
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


# ============ 分布式训练增强：DeepSpeed / FSDP 支持 ============

def setup_distributed_with_deepspeed(deepspeed_config_path=None, zero_stage=2):
    """
    初始化 DeepSpeed 分布式训练

    Args:
        deepspeed_config_path: DeepSpeed 配置文件路径（None 时自动生成）
        zero_stage: ZeRO 阶段（1/2/3）

    Returns:
        (local_rank, model, optimizer, deepspeed_engine)
    """
    import deepspeed
    local_rank = init_distributed_mode()

    if deepspeed_config_path is None:
        deepspeed_config_path = generate_deepspeed_config(zero_stage=zero_stage)

    return local_rank, deepspeed_config_path


def generate_deepspeed_config(zero_stage=2, offload=False, bf16=True):
    """
    生成 DeepSpeed 配置文件（返回 JSON 字符串和路径）
    """
    import json, tempfile

    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 50,
        "wall_clock_breakdown": False,
        "bf16": {"enabled": bf16},
        "fp16": {"enabled": not bf16},
        "zero_optimization": {
            "stage": zero_stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True,
            "round_robin_gradients": True,
            "offload_optimizer": {
                "device": "cpu" if offload else "none",
                "pin_memory": True
            } if offload else None,
            "offload_param": {
                "device": "cpu" if (offload and zero_stage == 3) else "none",
                "pin_memory": True
            } if (offload and zero_stage == 3) else None,
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": "auto"
            }
        },
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "warmup_min_ratio": 0.1,
                "warmup_num_steps": "auto",
                "total_num_steps": "auto"
            }
        },
        "activation_checkpointing": {
            "partition_activations": False,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False
        }
    }

    # 清理 None 字段
    if config["zero_optimization"]["offload_optimizer"] is None:
        del config["zero_optimization"]["offload_optimizer"]
    if config["zero_optimization"]["offload_param"] is None:
        del config["zero_optimization"]["offload_param"]

    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"ds_config_zero{zero_stage}.json")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    Logger(f"[DeepSpeed] 配置文件已生成: {config_path} (ZeRO-{zero_stage}, offload={offload})")
    return config_path


def wrap_model_with_deepspeed(model, optimizer, deepspeed_config_path):
    """
    用 DeepSpeed 包装模型和优化器
    """
    import deepspeed
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=deepspeed_config_path,
    )
    return model_engine, optimizer, scheduler


def setup_fsdp(rank, world_size, mixed_precision=True, sharding_strategy="FULL_SHARD"):
    """
    设置 FSDP（Fully Sharded Data Parallel）分布式训练

    Args:
        rank: 当前进程 rank
        world_size: 总进程数
        mixed_precision: 是否启用混合精度
        sharding_strategy: 分片策略（FULL_SHARD / SHARD_GRAD_OP / NO_SHARD）
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ) if mixed_precision else None

    return {
        "fsdp_class": FSDP,
        "mixed_precision": mp_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "sharding_strategy": sharding_strategy,
    }


def get_world_size():
    """获取分布式训练的 world size"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    """获取当前进程的 rank"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def all_reduce_mean(tensor):
    """对 tensor 做 all-reduce mean（分布式安全的指标聚合）"""
    if not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


def setup_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_params(model, ignore_patterns=None):
    """统计模型参数"""
    ignore_patterns = ignore_patterns or []
    def should_count(n):
        return not any(p in n for p in ignore_patterns)
    total = sum(p.numel() for n, p in model.named_parameters() if should_count(n)) / 1e6
    trainable = sum(p.numel() for n, p in model.named_parameters()
                    if should_count(n) and p.requires_grad) / 1e6
    Logger(f"Model Params: {total:.2f}M (Trainable: {trainable:.2f}M)")


class SkipBatchSampler(Sampler):
    """
    支持跳过指定 batch 数的采样器（用于断点续训）
    """

    def __init__(self, sampler, batch_size: int, skip_batches: int = 0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


def edu_sft_collate_fn(batch):
    """SFT 数据整理函数"""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])

    pixel_data = [b["pixel_values"] for b in batch]
    pixel_values = torch.stack(pixel_data)
    image_grid_thw = torch.stack([b["image_grid_thw"] for b in batch])

    return input_ids, attention_mask, labels, pixel_values, image_grid_thw


def edu_grpo_collate_fn(batch):
    """GRPO 数据整理函数"""
    prompt_input_ids = torch.stack([b["prompt_input_ids"] for b in batch])
    prompt_attention_mask = torch.stack([b["prompt_attention_mask"] for b in batch])
    gt_response_ids = torch.stack([b["gt_response_ids"] for b in batch])

    pixel_data = [b["pixel_values"] for b in batch]
    pixel_values = torch.stack(pixel_data)
    image_grid_thw = torch.stack([b["image_grid_thw"] for b in batch])

    return prompt_input_ids, prompt_attention_mask, gt_response_ids, pixel_values, image_grid_thw