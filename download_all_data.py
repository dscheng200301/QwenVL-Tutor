"""
QwenVL-Tutor 一键下载所有数据集
自动下载、转换训练数据集，并从训练集创建评估数据集

使用方法:
    python download_all_data.py              # 下载所有数据集并创建评估集
    python download_all_data.py --train     # 仅下载训练数据集
    python download_all_data.py --eval       # 仅创建评估数据集（需要先有训练集）
    python download_all_data.py --datasets scienceqa ceval  # 下载指定数据集
"""

import os
import sys
import argparse
import random
import subprocess
from pathlib import Path

# 确保当前目录在 path 中
SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR)

# 设置随机种子
random.seed(42)

# 设置 HuggingFace 环境
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# 数据集配置
# 格式: (dataset_name, train_output, eval_output, max_samples, eval_ratio, eval_max_samples, description)
DATASETS_CONFIG = [
    # === 核心中文图文数学数据集（新增） ===
    ("we_math", "dataset/edu_we_math.parquet", "dataset/eval/eval_we_math.parquet", 
     None, 0.05, 500, "We-Math 2.0 系统性数学知识体系"),
    
    ("geo170k", "dataset/edu_geo170k.parquet", "dataset/eval/eval_geo170k.parquet", 
     50000, 0.05, 500, "Geo170K 几何推理专项数据集"),
    
    ("cmm_math", "dataset/edu_cmm_math.parquet", "dataset/eval/eval_cmm_math.parquet", 
     None, 0.05, 500, "CMM-Math 中文K12数学图文数据集"),
    
    ("math_real", "dataset/edu_math_real.parquet", "dataset/eval/eval_math_real.parquet", 
     None, 0.10, 200, "MathReal 真实场景K12数学"),
    
    # === 核心图文数学数据集 ===
    ("scienceqa", "dataset/edu_science.parquet", "dataset/eval/eval_science.parquet", 
     None, 0.05, 500, "核心多模态科学问答数据集"),
    
    # === 中文图文数据集 ===
    ("windata-math", "dataset/edu_windata_math.parquet", "dataset/eval/eval_windata_math.parquet", 
     10000, 0.05, 500, "中文数学图文推理数据集"),
    
    # === OCR 和图表理解数据集 ===
    ("ocr_vqa", "dataset/edu_ocr.parquet", "dataset/eval/eval_ocr.parquet", 
     20000, 0.05, 500, "OCR 视觉问答数据集"),
    ("chartqa", "dataset/edu_chartqa.parquet", "dataset/eval/eval_chartqa.parquet", 
     10000, 0.05, 500, "图表问答数据集"),
    
    # === 中文理科数据集 ===
    ("ceval", "dataset/edu_ceval.parquet", "dataset/eval/eval_ceval.parquet", 
     None, 0.05, 200, "中文综合学科评估数据集"),
    ("cmmlu", "dataset/edu_cmmlu.parquet", "dataset/eval/eval_cmmlu.parquet", 
     None, 0.05, 500, "中文多学科多语言评估数据集"),
    
    # === 中文数学数据集 ===
    ("ape210k", "dataset/edu_ape210k.parquet", "dataset/eval/eval_ape210k.parquet", 
     20000, 0.05, 500, "中文小学数学应用题数据集"),
    ("openr1_math", "dataset/edu_openr1_math.parquet", "dataset/eval/eval_openr1_math.parquet", 
     20000, 0.05, 500, "中文 K12 数学推理链数据集"),
    ("gaokao_mathqa", "dataset/edu_gaokao_mathqa.parquet", "dataset/eval/eval_gaokao_mathqa.parquet", 
     None, 0.10, 100, "高考数学选择题数据集"),
    ("gaokao_mathcloze", "dataset/edu_gaokao_mathcloze.parquet", "dataset/eval/eval_gaokao_mathcloze.parquet", 
     None, 0.15, 50, "高考数学填空题数据集"),
    
    # === 数学推理数据集 ===
    ("mathverse", "dataset/edu_math_verse.parquet", "dataset/eval/eval_math_verse.parquet", 
     None, 0.05, 200, "数学视觉推理专业数据集"),
    ("mathvista", "dataset/edu_math_vista.parquet", "dataset/eval/eval_math_vista.parquet", 
     None, 0.10, 100, "数学视觉推理基准测试集"),
    
    # === 语言理解数据集 ===
    ("race", "dataset/edu_race.parquet", "dataset/eval/eval_race.parquet", 
     10000, 0.05, 500, "阅读理解数据集"),
]

def download_dataset(dataset_name, output_path, max_samples=None):
    """下载并转换单个训练数据集"""
    print(f"\n{'='*70}")
    print(f"📥 下载训练数据集: {dataset_name}")
    print(f"   输出路径: {output_path}")
    if max_samples:
        print(f"   样本限制: {max_samples}")
    print(f"{'='*70}")
    
    # 检查文件是否已存在
    if os.path.exists(output_path):
        print(f"✅ 文件已存在，跳过: {output_path}")
        return True
    
    # 构建命令
    cmd = [
        sys.executable,
        "scripts/convert_edu_data.py",
        "--dataset", dataset_name,
        "--output", output_path,
    ]
    
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])
    
    # 执行下载
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"✅ 成功: {dataset_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 失败: {dataset_name}")
        print(f"错误信息: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ 错误: 找不到 convert_edu_data.py 脚本")
        return False


def create_eval_set(train_path, eval_path, eval_ratio=0.05, eval_max_samples=500,
                    split_marker_col="split"):
    """
    从训练集中**真正分离**评估集（保证训练集和评估集完全不重叠）

    关键修复:
        原版本只采样写入 eval_path, 训练集仍含评估样本 → 数据泄露
        新版本: 评估样本从训练集中**彻底移除**, 保证评估有效性

    流程:
        1. 读取完整训练集
        2. 随机采样 N 个作为评估集
        3. 写评估集到 eval_path (含 split='eval' 标记)
        4. 写剩余训练集到 train_path (不含评估样本, 含 split='train' 标记)
    """
    import pyarrow.parquet as pq

    print(f"\n📊 分离评估集: {eval_path}")
    print(f"   训练集: {train_path}")
    print(f"   采样比例: {eval_ratio * 100}%")
    print(f"   最大样本数: {eval_max_samples}")

    # 检查文件是否存在
    if not os.path.exists(train_path):
        print(f"❌ 训练集不存在，跳过: {train_path}")
        return False

    # 检查评估集是否已存在
    if os.path.exists(eval_path):
        print(f"✅ 评估集已存在，跳过: {eval_path}")
        return True

    try:
        # 读取训练集
        table = pq.read_table(train_path)
        total = len(table)

        # 计算采样数量
        n = min(int(total * eval_ratio), eval_max_samples)
        n = max(n, 10)  # 至少 10 个样本
        n = min(n, total - 10)  # 至少留 10 个给训练

        print(f"   总样本数: {total}, 评估样本数: {n}, 训练样本数: {total - n}")

        # 随机采样（用固定种子保证可复现）
        all_indices = list(range(total))
        eval_indices = sorted(random.sample(all_indices, n))
        train_indices = sorted(set(all_indices) - set(eval_indices))

        # 评估表：含 split 标记
        eval_table = table.take(eval_indices)
        if split_marker_col not in eval_table.column_names:
            # 如果表里没有 split 列，添加
            import pyarrow as pa
            split_col = pa.array(["eval"] * len(eval_table), type=pa.string())
            eval_table = eval_table.append_column(split_marker_col, split_col)

        # 训练表：含 split 标记且**不含评估样本**
        train_table = table.take(train_indices)
        if split_marker_col not in train_table.column_names:
            import pyarrow as pa
            split_col = pa.array(["train"] * len(train_table), type=pa.string())
            train_table = train_table.append_column(split_marker_col, split_col)

        # 写评估集
        os.makedirs(os.path.dirname(eval_path), exist_ok=True)
        pq.write_table(eval_table, eval_path)

        # 写回训练集（不含评估样本）
        pq.write_table(train_table, train_path)

        print(f"✅ 成功:")
        print(f"   评估集: {eval_path} ({len(eval_table)} 条)")
        print(f"   训练集: {train_path} ({len(train_table)} 条, 已移除评估样本)")
        print(f"   🔒 训练集和评估集已严格分离，无数据泄露")
        return True
    except Exception as e:
        print(f"❌ 分离评估集失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="QwenVL-Tutor 一键下载所有数据集")
    parser.add_argument("--train", action="store_true", help="仅下载训练数据集")
    parser.add_argument("--eval", action="store_true", help="仅创建评估数据集（需要先有训练集）")
    parser.add_argument("--datasets", nargs="+", help="指定要下载的数据集名称")
    parser.add_argument("--max-workers", type=int, default=4, help="最大并行下载数")
    args = parser.parse_args()
    
    # 确保目录存在
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("dataset/eval", exist_ok=True)
    
    # 统计数据
    total = 0
    success = 0
    failed = []
    
    # 下载训练数据集并创建评估集
    if not args.eval:
        print("\n" + "="*70)
        print("🚀 开始下载训练数据集并创建评估集")
        print("="*70)
        
        for config in DATASETS_CONFIG:
            dataset_name = config[0]
            train_path = config[1]
            eval_path = config[2]
            max_samples = config[3]
            eval_ratio = config[4]
            eval_max_samples = config[5]
            description = config[6]
            
            # 如果指定了数据集，只下载指定的数据集
            if args.datasets and dataset_name not in args.datasets:
                continue
            
            total += 1
            print(f"\n📦 {total}. {dataset_name} - {description}")
            
            # 下载训练集
            if download_dataset(dataset_name, train_path, max_samples):
                success += 1
                # 创建评估集
                create_eval_set(train_path, eval_path, eval_ratio, eval_max_samples)
            else:
                failed.append(dataset_name)
    
    # 仅创建评估数据集
    if not args.train:
        print("\n" + "="*70)
        print("📊 创建评估数据集（仅从现有训练集采样）")
        print("="*70)
        
        for config in DATASETS_CONFIG:
            dataset_name = config[0]
            train_path = config[1]
            eval_path = config[2]
            eval_ratio = config[4]
            eval_max_samples = config[5]
            description = config[6]
            
            # 如果指定了数据集，只处理指定的数据集
            if args.datasets and dataset_name not in args.datasets:
                continue
            
            total += 1
            print(f"\n📦 {total}. {dataset_name} - {description}")
            
            if create_eval_set(train_path, eval_path, eval_ratio, eval_max_samples):
                success += 1
            else:
                failed.append(f"{dataset_name} (eval)")
    
    # 输出统计
    print("\n" + "="*70)
    print("📊 完成统计")
    print("="*70)
    print(f"总数据集数: {total}")
    print(f"成功: {success}")
    print(f"失败: {len(failed)}")
    
    if failed:
        print(f"\n⚠️ 失败的数据集:")
        for name in failed:
            print(f"  - {name}")
        print("\n💡 提示: 可以单独重试失败的数据集:")
        print(f"   python download_all_data.py --datasets {' '.join(failed)}")
    else:
        print("\n🎉 全部下载成功！")
    
    print("="*70)
    
    # 输出下一步提示
    if success > 0:
        print("\n📋 下一步:")
        print("  1. 查看下载的数据集: ls dataset/")
        print("  2. 查看评估数据集: ls dataset/eval/")
        print("  3. 开始训练: python trainer/train_sft.py")
        print("  4. 运行评估: python eval_edu.py --eval_all")


if __name__ == "__main__":
    main()
