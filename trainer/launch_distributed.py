"""
QwenVL-Tutor 分布式训练启动器

提供统一的启动接口，支持多种分布式方案:
    - 单卡训练 (单进程)
    - DDP 数据并行 (torchrun)
    - DeepSpeed ZeRO-1/2/3 (deepspeed launcher)
    - FSDP 全分片 (torchrun + accelerate)
    - Accelerate Launch (通用)

使用方法:
    # 单卡
    python trainer/launch_distributed.py --mode single

    # DDP 4 卡
    python trainer/launch_distributed.py --mode ddp --nproc_per_node 4

    # DeepSpeed ZeRO-2 4 卡
    python trainer/launch_distributed.py --mode deepspeed --nproc_per_node 4 --zero_stage 2

    # FSDP 4 卡
    python trainer/launch_distributed.py --mode fsdp --nproc_per_node 4

    # Accelerate
    python trainer/launch_distributed.py --mode accelerate --config configs/accelerate_ds.yaml
"""
import os
import sys
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)


def launch_single(args):
    """单卡训练"""
    cmd = [
        sys.executable, "trainer/train_sft.py",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
    ]
    if args.use_lora == 0:
        cmd += ["--use_lora", "0"]
    if args.save_weight:
        cmd += ["--save_weight", args.save_weight]
    if args.data_paths:
        cmd += ["--data_paths", args.data_paths]
    print(f"[单卡] 执行: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def launch_ddp(args):
    """DDP 数据并行（torchrun）"""
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--nnodes={args.nnodes}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        "trainer/train_sft.py",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
    ]
    if args.data_paths:
        cmd += ["--data_paths", args.data_paths]
    if args.save_weight:
        cmd += ["--save_weight", args.save_weight]
    print(f"[DDP] 执行: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def launch_deepspeed(args):
    """DeepSpeed ZeRO 训练"""
    # 先生成 DeepSpeed 配置
    from trainer.trainer_utils import generate_deepspeed_config
    ds_config_path = generate_deepspeed_config(
        zero_stage=args.zero_stage,
        offload=bool(args.deepspeed_offload),
        bf16=True,
    )

    if args.deepspeed_offload or args.zero_stage == 3:
        # 使用 deepspeed launcher 自带的 offload 支持
        cmd = [
            "deepspeed",
            f"--num_gpus={args.nproc_per_node}",
            f"--num_nodes={args.nnodes}",
            f"--master_addr={args.master_addr}",
            f"--master_port={args.master_port}",
            "trainer/train_sft.py",
            "--use_deepspeed", "1",
            "--deepspeed_zero_stage", str(args.zero_stage),
            "--deepspeed_offload", str(args.deepspeed_offload),
            "--deepspeed_config", ds_config_path,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
        ]
    else:
        # 简单 ZeRO 用 torchrun
        cmd = [
            "torchrun",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--nnodes={args.nnodes}",
            f"--master_addr={args.master_addr}",
            f"--master_port={args.master_port}",
            "trainer/train_sft.py",
            "--use_deepspeed", "1",
            "--deepspeed_zero_stage", str(args.zero_stage),
            "--deepspeed_config", ds_config_path,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
        ]

    if args.data_paths:
        cmd += ["--data_paths", args.data_paths]
    if args.save_weight:
        cmd += ["--save_weight", args.save_weight]

    print(f"[DeepSpeed ZeRO-{args.zero_stage}] 执行: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def launch_fsdp(args):
    """FSDP 全分片（torchrun）"""
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--nnodes={args.nnodes}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        "trainer/train_sft.py",
        "--use_fsdp", "1",
        "--gradient_checkpointing", "1",  # FSDP 通常需要梯度检查点
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
    ]
    if args.data_paths:
        cmd += ["--data_paths", args.data_paths]
    if args.save_weight:
        cmd += ["--save_weight", args.save_weight]
    print(f"[FSDP] 执行: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def launch_accelerate(args):
    """Accelerate Launch"""
    if not args.config:
        # 默认 ZeRO-2 配置
        default_config = os.path.join(PROJECT_ROOT, "configs", "accelerate_default.yaml")
        if not os.path.exists(default_config):
            os.makedirs(os.path.dirname(default_config), exist_ok=True)
            with open(default_config, "w") as f:
                f.write("""compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  zero_optimization:
    stage: 2
  bf16: true
mixed_precision: bf16
num_processes: 4
""")
        args.config = default_config

    cmd = [
        "accelerate", "launch",
        "--config_file", args.config,
        "--num_processes", str(args.nproc_per_node),
        "--num_machines", str(args.nnodes),
        "trainer/train_sft.py",
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
    ]
    if args.data_paths:
        cmd += ["--data_paths", args.data_paths]
    if args.save_weight:
        cmd += ["--save_weight", args.save_weight]
    print(f"[Accelerate] 执行: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def main():
    parser = argparse.ArgumentParser(description="QwenVL-Tutor 分布式训练启动器")
    parser.add_argument("--mode", type=str, default="ddp",
                        choices=["single", "ddp", "deepspeed", "fsdp", "accelerate"],
                        help="训练模式")
    parser.add_argument("--nproc_per_node", type=int, default=1, help="每节点 GPU 数")
    parser.add_argument("--nnodes", type=int, default=1, help="总节点数")
    parser.add_argument("--node_rank", type=int, default=0, help="当前节点 rank")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1", help="master 地址")
    parser.add_argument("--master_port", type=str, default="29500", help="master 端口")

    # DeepSpeed 特定参数
    parser.add_argument("--zero_stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--deepspeed_offload", type=int, default=0, choices=[0, 1])

    # Accelerate 特定参数
    parser.add_argument("--config", type=str, default=None, help="accelerate 配置文件")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--use_lora", type=int, default=1, choices=[0, 1])
    parser.add_argument("--data_paths", type=str, default=None)
    parser.add_argument("--save_weight", type=str, default="edu_sft")

    args = parser.parse_args()

    print("=" * 70)
    print(f"QwenVL-Tutor 分布式训练启动器")
    print("=" * 70)
    print(f"模式: {args.mode}")
    print(f"GPU 数: {args.nproc_per_node} × 节点 {args.nnodes} = {args.nproc_per_node * args.nnodes}")
    if args.mode == "deepspeed":
        print(f"ZeRO Stage: {args.zero_stage}, Offload: {bool(args.deepspeed_offload)}")
    print("=" * 70)

    if args.mode == "single":
        return launch_single(args)
    elif args.mode == "ddp":
        return launch_ddp(args)
    elif args.mode == "deepspeed":
        return launch_deepspeed(args)
    elif args.mode == "fsdp":
        return launch_fsdp(args)
    elif args.mode == "accelerate":
        return launch_accelerate(args)


if __name__ == "__main__":
    sys.exit(main())
