"""
QwenSearch GRPO 强化数据准备脚本

从多个 SFT 数据集中采样并构建 GRPO 强化训练数据。

GRPO 数据集设计原则（5,000 条高质量强化样本）:
    - 选择有 GT 答案、便于奖励模型评估的: CMMU、MMSciBench、MathVerse、MathVista、ScienceQA
    - 训练流程: SFT → GRPO（不再使用 DPO，避免错误偏好信号）

用法:
    # 默认输出到 dataset/edu_grpo.parquet
    python scripts/optimize/build_preference_data.py

    # 指定输出路径
    python scripts/optimize/build_preference_data.py --output dataset/edu_grpo.parquet
"""
import os
import sys
import argparse
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR.parent.parent)


# GRPO 阶段精选数据集（有 GT 答案，便于奖励模型评估）
GRPO_DATASETS = {
    'cmmu': {
        'path': 'dataset/edu_cmmu.parquet',
        'max_samples': 1500,
        'note': '中文 K12 7 门学科，多学科可全面优化',
    },
    'mmscibench': {
        'path': 'dataset/edu_mmscibench.parquet',
        'max_samples': 1000,
        'note': '中学物理数学，需详细推理，奖励信号强',
    },
    'math_verse': {
        'path': 'dataset/edu_math_verse.parquet',
        'max_samples': 1000,
        'note': '数学视觉推理，关键词匹配评估准',
    },
    'math_vista': {
        'path': 'dataset/edu_math_vista.parquet',
        'max_samples': 500,
        'note': '视觉推理基准，答案匹配评估',
    },
    'scienceqa': {
        'path': 'dataset/edu_science.parquet',
        'max_samples': 1000,
        'note': '完整推理链，五维度奖励清晰',
    },
}


def read_parquet(path):
    """读取 parquet 文件"""
    import pyarrow.parquet as pq
    if not os.path.exists(path):
        print(f"  [WARN] 文件不存在: {path}，跳过")
        return None
    return pq.read_table(path)


def sample_table(table, max_samples):
    """从 Table 中随机采样"""
    n = len(table)
    if n <= max_samples:
        return table
    indices = sorted(random.sample(range(n), max_samples))
    return table.take(indices)


def build_grpo_data(output_path):
    """构建 GRPO 强化数据"""
    import pyarrow as pa

    print("=" * 70)
    print("🔨 构建 GRPO 强化数据")
    print("=" * 70)
    print(f"目标规模: ~{sum(d['max_samples'] for d in GRPO_DATASETS.values())} 条")
    print(f"数据集数量: {len(GRPO_DATASETS)}")
    print()

    all_tables = []
    total_target = 0
    for name, cfg in GRPO_DATASETS.items():
        print(f"📦 加载 {name}: {cfg['path']}")
        print(f"   目标: {cfg['max_samples']} 条 | {cfg['note']}")
        table = read_parquet(cfg['path'])
        if table is None:
            continue
        before = len(table)
        table = sample_table(table, cfg['max_samples'])
        all_tables.append(table)
        total_target += cfg['max_samples']
        print(f"   原始: {before} -> 采样: {len(table)}\n")

    if not all_tables:
        print("❌ 没有可用数据，请先运行: python scripts/download_all_data.py")
        return

    merged = pa.concat_tables(all_tables)
    print("=" * 70)
    print(f"✅ 合并后总数: {len(merged)} 条")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    import pyarrow.parquet as pq
    pq.write_table(merged, output_path)
    print(f"💾 已保存到: {output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="QwenSearch GRPO 强化数据准备")
    parser.add_argument("--output", type=str,
                        default='dataset/edu_grpo.parquet',
                        help="GRPO 输出路径")
    args = parser.parse_args()

    build_grpo_data(args.output)

    print("\n" + "=" * 70)
    print("🎉 GRPO 强化数据准备完成！")
    print("=" * 70)
    print("下一步训练:")
    print("  python trainer/train_grpo.py --from_weight ../out/edu_sft --epochs 1")


if __name__ == "__main__":
    main()
