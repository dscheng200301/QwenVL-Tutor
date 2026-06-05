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
    """初始化分布式训练模式"""
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


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


def edu_dpo_collate_fn(batch):
    """DPO 数据整理函数"""
    chosen_input_ids = torch.stack([b["chosen_input_ids"] for b in batch])
    chosen_attention_mask = torch.stack([b["chosen_attention_mask"] for b in batch])
    chosen_labels = torch.stack([b["chosen_labels"] for b in batch])

    rejected_input_ids = torch.stack([b["rejected_input_ids"] for b in batch])
    rejected_attention_mask = torch.stack([b["rejected_attention_mask"] for b in batch])
    rejected_labels = torch.stack([b["rejected_labels"] for b in batch])

    pixel_data = [b["pixel_values"] for b in batch]
    pixel_values = torch.stack(pixel_data)
    image_grid_thw = torch.stack([b["image_grid_thw"] for b in batch])

    return (chosen_input_ids, chosen_attention_mask, chosen_labels,
            rejected_input_ids, rejected_attention_mask, rejected_labels,
            pixel_values, image_grid_thw)


def edu_grpo_collate_fn(batch):
    """GRPO 数据整理函数"""
    prompt_input_ids = torch.stack([b["prompt_input_ids"] for b in batch])
    prompt_attention_mask = torch.stack([b["prompt_attention_mask"] for b in batch])
    gt_response_ids = torch.stack([b["gt_response_ids"] for b in batch])

    pixel_data = [b["pixel_values"] for b in batch]
    pixel_values = torch.stack(pixel_data)
    image_grid_thw = torch.stack([b["image_grid_thw"] for b in batch])

    return prompt_input_ids, prompt_attention_mask, gt_response_ids, pixel_values, image_grid_thw