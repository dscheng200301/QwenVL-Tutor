"""
QwenSearch 评估结果对比工具

用法:
    # 对比最新两次评估
    python compare_evals.py
    
    # 指定两个文件对比
    python compare_evals.py --file1 eval_results/sft_20250101_120000.json --file2 eval_results/sft_20250102_120000.json

输出:
    - 各指标的绝对/相对变化
    - 能力变化摘要（上升 ✅ / 下降 ⚠️）
"""
import os
import sys
import json
import argparse
import glob


def load_results(path: str) -> dict:
    """加载评估结果 JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_latest_files() -> tuple:
    """自动找到最新的两个评估文件"""
    files = sorted(glob.glob("eval_results/*.json"))
    # 排除 baseline.json 和 latest.json
    files = [f for f in files if not f.endswith("baseline.json") and not f.endswith("latest.json")]
    if len(files) < 2:
        print("❌ 需要至少 2 个评估结果文件才能对比")
        print(f"   当前找到 {len(files)} 个文件")
        sys.exit(1)
    return files[-2], files[-1]


def compare_results(r1: dict, r2: dict):
    """逐指标对比两次结果"""
    keys = set(list(r1.keys()) + list(r2.keys()))
    # 过滤出数值型指标
    numeric_keys = [k for k in keys if isinstance(r1.get(k), (int, float)) or isinstance(r2.get(k), (int, float))]
    meta_keys = [k for k in keys if k not in numeric_keys]
    
    print("\n" + "=" * 70)
    print("📊 评估结果对比分析")
    print("=" * 70)
    
    # 元信息
    print(f"\n📋 版本信息:")
    print(f"  文件1: {r1.get('timestamp', 'N/A')} | stage={r1.get('stage', 'N/A')} | model={r1.get('model_path', 'N/A')}")
    print(f"  文件2: {r2.get('timestamp', 'N/A')} | stage={r2.get('stage', 'N/A')} | model={r2.get('model_path', 'N/A')}")
    
    # 指标对比
    print(f"\n📈 指标变化:")
    print(f"  {'指标':<25s} {'文件1':>10s} {'文件2':>10s} {'变化':>10s} {'趋势'}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    
    improved = 0
    regressed = 0
    
    for key in sorted(numeric_keys):
        v1 = r1.get(key, 0)
        v2 = r2.get(key, 0)
        if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
            continue
        delta = v2 - v1
        if abs(delta) < 0.001:
            trend = "➡️ 持平"
        elif delta > 0:
            trend = "✅ 上升"
            improved += 1
        else:
            trend = "⚠️ 下降"
            regressed += 1
        
        print(f"  {key:<25s} {v1:>10.4f} {v2:>10.4f} {delta:>+10.4f} {trend}")
    
    # 总结
    print(f"\n📊 总结: {improved} 项上升, {regressed} 项下降")
    
    if regressed > improved:
        print(f"  ⚠️ 退化指标多于改善指标，建议检查训练配置")
    elif improved > regressed:
        print(f"  ✅ 模型能力整体提升")
    else:
        print(f"  ➡️ 模型能力无显著变化")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比两次评估结果")
    parser.add_argument("--file1", type=str, default=None, help="第一个评估文件")
    parser.add_argument("--file2", type=str, default=None, help="第二个评估文件")
    args = parser.parse_args()
    
    if args.file1 and args.file2:
        f1, f2 = args.file1, args.file2
    else:
        f1, f2 = find_latest_files()
    
    print(f"📁 对比文件:")
    print(f"  文件1: {f1}")
    print(f"  文件2: {f2}")
    
    r1 = load_results(f1)
    r2 = load_results(f2)
    compare_results(r1, r2)