"""
QwenSearch 数据重采样工具 — 根据评估反馈动态调整数据集权重

原理：
    1. 读取最近一次 --eval_all 评估结果
    2. 计算每个数据集的"能力得分"（关键词匹配率）
    3. 得分低的数据集 → 提高采样权重（弱项多练）
    4. 输出推荐的 --data_paths 和权重配置

用法:
    # 基于最新评估结果生成重采样建议
    python scripts/resample_data.py
    
    # 指定评估结果文件
    python scripts/resample_data.py --eval_file eval_results/sft_20250101_120000.json
    
    # 输出到文件（供 train_sft.py 使用）
    python scripts/resample_data.py --output weights.json
"""
import os
import sys
import json
import argparse
import glob


# 训练数据集文件路径（含图/纯文本标记）
TRAIN_DATASETS = {
    'scienceqa': 'dataset/edu_science.parquet',
    'ceval': 'dataset/edu_ceval.parquet',
    'ocr': 'dataset/edu_ocr.parquet',
    'ape210k': 'dataset/edu_ape210k.parquet',
    'chartqa': 'dataset/edu_chartqa.parquet',
    'cmmlu': 'dataset/edu_cmmlu.parquet',
    'math_verse': 'dataset/edu_math_verse.parquet',
    'math_vista': 'dataset/edu_math_vista.parquet',
    'race': 'dataset/edu_race.parquet',
    'openr1_math': 'dataset/edu_openr1_math.parquet',
    'gaokao_mathqa': 'dataset/edu_gaokao_mathqa.parquet',
    'gaokao_mathcloze': 'dataset/edu_gaokao_mathcloze.parquet',
}

# 评估数据集到训练数据集的映射
EVAL_TO_TRAIN = {
    'scienceqa': 'scienceqa',
    'ceval': 'ceval',
    'ocr': 'ocr',
    'ape210k': 'ape210k',
    'chartqa': 'chartqa',
    'cmmlu': 'cmmlu',
    'math_verse': 'math_verse',
    'math_vista': 'math_vista',
    'race': 'race',
    'openr1_math': 'openr1_math',
    'gaokao_mathqa': 'gaokao_mathqa',
    'gaokao_mathcloze': 'gaokao_mathcloze',
}


def load_eval_results(path: str) -> dict:
    """加载评估结果"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_latest_eval() -> str:
    """自动找最新的评估结果文件"""
    files = sorted(glob.glob("eval_results/*.json"))
    files = [f for f in files if not f.endswith("baseline.json") and not f.endswith("latest.json")]
    if not files:
        print("❌ 未找到评估结果文件，请先运行: python eval_edu.py --stage sft --eval_all")
        sys.exit(1)
    return files[-1]


def compute_resample_weights(eval_path: str, min_weight: float = 0.3, max_weight: float = 3.0):
    """
    根据评估结果计算重采样权重
    
    公式:
        基础权重 = 1.0
        调整系数 = (0.5 / max(score, 0.01))  # 分低权重高
        最终权重 = clamp(基础权重 × 调整系数, min_weight, max_weight)
    """
    eval_data = load_eval_results(eval_path)
    
    # 收集各数据集的得分
    scores = {}
    for key, value in eval_data.items():
        if key.startswith("accuracy_") or key.startswith("kw_match_"):
            dataset_name = key.replace("accuracy_", "").replace("kw_match_", "")
            if dataset_name in TRAIN_DATASETS:
                scores[dataset_name] = float(value)
    
    if not scores:
        print("⚠️ 评估结果中未找到数据集得分，使用均匀权重")
        return {k: 1.0 for k in TRAIN_DATASETS}
    
    # 计算权重
    weights = {}
    print("\n" + "=" * 70)
    print("📊 基于评估反馈的重采样权重")
    print("=" * 70)
    print(f"  {'数据集':<20s} {'得分':>8s} {'权重':>8s} {'建议'}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10}")
    
    total_weight = 0
    for ds_name, ds_path in TRAIN_DATASETS.items():
        if not os.path.exists(ds_path):
            continue
        
        score = scores.get(ds_name, 0.5)  # 未评估过的默认 0.5
        # 分低 → 权重高（弱项多练）
        adjust = 0.5 / max(score, 0.01)
        weight = max(min_weight, min(max_weight, adjust))
        weights[ds_name] = round(weight, 2)
        total_weight += weight
        
        hint = ""
        if weight > 2.0:
            hint = "🔴 重点训练"
        elif weight > 1.5:
            hint = "🟡 加强训练"
        elif weight < 0.5:
            hint = "🟢 可适当减少"
        
        print(f"  {ds_name:<20s} {score:>8.3f} {weight:>8.2f} {hint}")
    
    print(f"\n  总权重: {total_weight:.2f} (均匀时为 {len(weights):.1f})")
    
    return weights


def generate_data_paths(weights: dict) -> str:
    """生成带权重的 --data_paths 参数字符串"""
    paths = []
    for ds_name, weight in sorted(weights.items()):
        if weight <= 0:
            continue
        paths.append(TRAIN_DATASETS[ds_name])
    return ",".join(paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估反馈数据重采样")
    parser.add_argument("--eval_file", type=str, default=None, help="评估结果 JSON 文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出权重 JSON 文件")
    args = parser.parse_args()
    
    eval_path = args.eval_file or find_latest_eval()
    print(f"📁 加载评估结果: {eval_path}")
    
    weights = compute_resample_weights(eval_path)
    data_paths = generate_data_paths(weights)
    
    print(f"\n📋 推荐的训练命令:")
    print(f"  python trainer/train_sft.py --data_paths \"{data_paths}\"")
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                "eval_source": eval_path,
                "weights": weights,
                "data_paths": data_paths,
            }, f, ensure_ascii=False, indent=2)
        print(f"📁 权重已导出: {args.output}")