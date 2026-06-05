"""
QwenSearch 评估报告自动生成工具

功能:
    1. 从多个评估结�?JSON 生成 Markdown 报告
    2. 自动绘制能力雷达�?    3. 自动绘制各数据集准确率柱状图
    4. 自动生成变化趋势摘要

用法:
    # 基于最新评估生成报�?    python scripts/generate_report.py

    # 指定评估文件
    python scripts/generate_report.py --eval_files eval_results/sft_xxx.json

    # 对比模式：生�?baseline vs final 对比报告
    python scripts/generate_report.py --mode compare --baseline eval_results/baseline_xxx.json --final eval_results/sft_xxx.json

    # 输出到指定文�?    python scripts/generate_report.py --output report.md
"""
import os
import sys
import json
import argparse
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR.parent.parent)


def load_results(path: str) -> dict:
    """加载评估结果"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_latest_eval() -> str:
    """找最新的评估结果文件"""
    files = sorted(glob.glob("eval_results/*.json"))
    files = [f for f in files if not f.endswith("latest.json") and not f.endswith("audit_samples.json")]
    if not files:
        print("�?未找到评估结�?)
        sys.exit(1)
    return files[-1]


def get_dataset_metrics(eval_data: dict) -> dict:
    """从评估结果中提取各数据集的核心指�?""
    metrics = {}
    if "datasets" in eval_data:
        for ds_name, ds in eval_data["datasets"].items():
            acc = ds.get("accuracy")
            step = ds.get("step_completeness")
            scaffold = ds.get("scaffolding_rate")
            metrics[ds_name] = {
                "accuracy": acc,
                "step_completeness": step,
                "scaffolding_rate": scaffold,
                "total": ds.get("total", 0),
            }
    return metrics


def _bar(value: float, max_value: float = 1.0, width: int = 20) -> str:
    """生成进度条字符串"""
    if value is None:
        return "�? * width
    bar_len = int(value / max_value * width)
    return "�? * bar_len + "�? * (width - bar_len)


def _format_pct(v: float) -> str:
    """格式化百分比"""
    if v is None:
        return "N/A"
    return f"{v*100:.1f}%"


def make_default_filename(stage: str = "eval", avg_accuracy: float = None,
                          ext: str = "md") -> str:
    """
    生成默认报告文件名（含时间戳 + 效果）

    格式: {stage}_{YYYYMMDD}_{HHMM}_acc{X.XXXX}.{ext}
    示例: sft_20260605_1430_acc0.6123.md
    """
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M")
    if avg_accuracy is not None:
        return f"{stage}_{ts}_acc{avg_accuracy:.4f}.{ext}"
    return f"{stage}_{ts}.{ext}"


def get_avg_accuracy(eval_files: list) -> float:
    """从评估文件中计算平均准确率"""
    if not eval_files:
        return None
    try:
        results = [load_results(f) for f in eval_files if os.path.exists(f)]
        if not results:
            return None
        latest = results[-1]
        metrics = get_dataset_metrics(latest)
        scores = [m["score"] for m in metrics.values() if "score" in m and isinstance(m["score"], (int, float))]
        if not scores:
            return None
        return sum(scores) / len(scores)
    except Exception:
        return None


def generate_markdown_report(eval_files: list, output_path: str = None,
                              stage: str = "eval",
                              include_plots: bool = True) -> str:
    """生成 Markdown 评估报告
    Args:
        eval_files: 评估结果文件列表
        output_path: 输出路径（None 时自动生成带时间戳和效果的文件名）
        stage: 评估阶段（用于默认文件名）
    """
    print(f"📊 生成报告，评估文�? {len(eval_files)} �?)

    # 加载所有评�?    results = [load_results(f) for f in eval_files]

    # 报告内容
    md = []
    md.append("# QwenSearch 评估报告\n")
    md.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"**评估文件�?*: {len(eval_files)}")
    md.append("")

    # 1. 概览
    md.append("## 1. 概览\n")
    for i, (f, r) in enumerate(zip(eval_files, results), 1):
        ts = r.get("timestamp", "N/A")
        stage = r.get("stage", "N/A")
        model = r.get("model_path", "N/A")
        md.append(f"### 评估 {i}: {stage}")
        md.append(f"- **时间�?*: {ts}")
        md.append(f"- **模型**: `{model}`")
        md.append(f"- **文件**: `{f}`")
        md.append("")

    # 2. 数据集级指标
    md.append("## 2. 数据集级指标（最新一次评估）\n")
    latest = results[-1]
    latest_metrics = get_dataset_metrics(latest)

    if latest_metrics:
        md.append("| 数据�?| 准确�?| 步骤�?| 引导�?| 样本�?| 可视�?|")
        md.append("|--------|:------:|:------:|:------:|:------:|--------|")
        for ds_name, m in sorted(latest_metrics.items()):
            acc = _format_pct(m["accuracy"])
            step = _format_pct(m["step_completeness"])
            scaf = _format_pct(m["scaffolding_rate"])
            total = m["total"]
            bar = _bar(m["accuracy"])
            md.append(f"| {ds_name} | {acc} | {step} | {scaf} | {total} | `{bar}` |")
        md.append("")

    # 3. 对比分析（如果有多个文件�?    if len(results) >= 2:
        md.append("## 3. 对比分析\n")
        baseline = results[0]
        final = results[-1]
        base_metrics = get_dataset_metrics(baseline)
        final_metrics = get_dataset_metrics(final)

        common = set(base_metrics.keys()) & set(final_metrics.keys())
        if common:
            md.append("| 数据�?| 基线准确�?| 最终准确率 | 变化 | 趋势 |")
            md.append("|--------|:----------:|:----------:|:----:|------|")
            improved = 0
            regressed = 0
            for ds in sorted(common):
                base_acc = base_metrics[ds]["accuracy"] or 0
                final_acc = final_metrics[ds]["accuracy"] or 0
                delta = final_acc - base_acc
                if abs(delta) < 0.01:
                    trend = "�?持平"
                elif delta > 0:
                    trend = "�?上升"
                    improved += 1
                else:
                    trend = "⚠️ 下降"
                    regressed += 1
                md.append(f"| {ds} | {_format_pct(base_acc)} | {_format_pct(final_acc)} | {delta:+.4f} | {trend} |")
            md.append("")
            md.append(f"**总结**: {improved} 项上�? {regressed} 项下�?)
            md.append("")

    # 4. 弱项数据�?    md.append("## 4. 弱项数据集（需要重点训练）\n")
    if latest_metrics:
        sorted_by_acc = sorted(
            [(ds, m) for ds, m in latest_metrics.items() if m["accuracy"] is not None],
            key=lambda x: x[1]["accuracy"]
        )
        md.append("| 排名 | 数据�?| 准确�?| 建议 |")
        md.append("|:----:|--------|:------:|------|")
        for i, (ds, m) in enumerate(sorted_by_acc[:5], 1):
            acc = m["accuracy"]
            if acc < 0.3:
                suggestion = "🔴 重点补充训练数据"
            elif acc < 0.5:
                suggestion = "🟡 加强训练"
            else:
                suggestion = "�?保持"
            md.append(f"| {i} | {ds} | {_format_pct(acc)} | {suggestion} |")
        md.append("")

    # 5. 五维度细粒度（如果有�?    if "fine_grained" in latest:
        md.append("## 5. 五维度细粒度评分\n")
        fg = latest["fine_grained"]
        md.append("| 维度 | 得分 |")
        md.append("|------|:----:|")
        for dim in ["accuracy", "completeness", "fluency", "scaffolding", "format"]:
            if dim in fg:
                score = fg[dim]
                md.append(f"| {dim} | {score:.3f} |")
        md.append("")

    # 6. 置信区间（如果有�?    if any("confidence_intervals" in ds for ds in latest_metrics.values()):
        md.append("## 6. 置信区间�?5% Bootstrap）\n")
        md.append("| 数据�?| 指标 | 区间 |")
        md.append("|--------|------|------|")
        for ds_name, m in latest_metrics.items():
            ds_full = latest["datasets"].get(ds_name, {})
            ci = ds_full.get("confidence_intervals", {})
            for metric, ci_data in ci.items():
                lo = ci_data.get("lower", 0)
                hi = ci_data.get("upper", 0)
                md.append(f"| {ds_name} | {metric} | [{lo:.3f}, {hi:.3f}] |")
        md.append("")

    # 7. 建议
    md.append("## 7. 后续建议\n")
    if latest_metrics:
        weak_count = sum(1 for m in latest_metrics.values()
                          if m["accuracy"] is not None and m["accuracy"] < 0.5)
        if weak_count > 0:
            md.append(f"- 🚨 **�?{weak_count} 个数据集准确�?< 50%**，建议使�?`resample_data.py` 重新生成训练权重")
        if len(results) >= 2:
            md.append("- 📈 **建议运行** `python scripts/compare_evals.py --show_weak_datasets` 查看详细对比")
        md.append("- 🔍 **建议运行** `python scripts/analyze_errors.py` 分析错误类型")
        md.append("- 📊 **建议运行** `python scripts/meta_evaluation.py --check_consistency` 检查指标一致�?)
        md.append("")

    # 保存
    report_md = "\n".join(md)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_md)
    print(f"💾 报告已保�? {output_path}")
    print(f"   总行�? {len(md)}")
    return report_md


def main():
    parser = argparse.ArgumentParser(description="QwenSearch 评估报告生成")
    parser.add_argument("--eval_files", nargs="+", default=None,
                        help="评估结果文件列表（默认使用最新的）")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "compare"],
                        help="报告模式：single=单次评估，compare=对比")
    parser.add_argument("--baseline", type=str, default=None, help="对比模式的基线文件")
    parser.add_argument("--final", type=str, default=None, help="对比模式的最终文件")
    parser.add_argument("--output", type=str, default=None,
                        help="报告输出路径（None 时自动生成 {stage}_YYYYMMDD_HHMM_accX.XXXX.md）")
    parser.add_argument("--stage", type=str, default="eval",
                        help="评估阶段（用于默认文件名）")
    args = parser.parse_args()

    # 确定评估文件
    if args.mode == "compare":
        if not (args.baseline and args.final):
            print("对比模式需要指定 --baseline 和 --final")
            sys.exit(1)
        eval_files = [args.baseline, args.final]
    elif args.eval_files:
        eval_files = args.eval_files
    else:
        eval_files = [find_latest_eval()]

    # 如果未指定输出路径，自动生成带时间戳和效果的文件名
    if args.output is None:
        avg_acc = get_avg_accuracy(eval_files)
        # 尝试从最新评估结果中获取 stage
        stage = args.stage
        try:
            if os.path.exists(eval_files[-1]):
                with open(eval_files[-1], "r", encoding="utf-8") as f:
                    data = json.load(f)
                stage = data.get("stage", args.stage)
        except Exception:
            pass
        os.makedirs("eval_results", exist_ok=True)
        filename = make_default_filename(stage=stage, avg_accuracy=avg_acc)
        output_path = os.path.join("eval_results", filename)
        print(f"自动生成报告文件名: {filename}")
    else:
        output_path = args.output

    generate_markdown_report(eval_files, output_path, stage=args.stage)


if __name__ == "__main__":
    main()
