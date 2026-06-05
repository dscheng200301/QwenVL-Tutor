"""
QwenSearch 一站式评估脚本

合并功能（替代 8 个分散脚本）:
    run       - 运行评估（替代 eval_edu.py）
    compare   - 对比两次评估（替代 compare_evals.py）
    errors    - 错误案例分析（替代 analyze_errors.py）
    meta      - 元评估（替代 meta_evaluation.py）
    report    - 生成评估报告（替代 generate_report.py）
    all       - 一体化：run + meta + report

🆕 自动加速集成:
    - run/all 默认使用 vLLM 推理（5-20x 加速）
    - 自动检测 GPU 数量，决定 TP（张量并行）数
    - vLLM 不可用时自动降级到 HF transformers

向后兼容:
    旧脚本（eval_edu.py, compare_evals.py 等）仍可独立运行
    本脚本通过子命令提供统一入口

用法:
    # 基础评估（自动 vLLM）
    python scripts/edu_evaluate.py run --stage sft --model_path out/edu_sft --eval_all

    # 强制 HF 后端
    python scripts/edu_evaluate.py run --stage sft --model_path out/edu_sft --no_vllm

    # 训练前基线
    python scripts/edu_evaluate.py run --stage baseline --eval_all --max_samples 200

    # 对比两次评估
    python scripts/edu_evaluate.py compare --show_weak

    # 错误归类
    python scripts/edu_evaluate.py errors --output_errors errors.json

    # 元评估
    python scripts/edu_evaluate.py meta --check_consistency

    # 生成报告
    python scripts/edu_evaluate.py report --output report.md

    # 一体化（推荐）
    python scripts/edu_evaluate.py all --stage sft --model_path out/edu_sft --eval_all
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR.parent.parent)


# ============================================================================
# 🆕 自动加速：GPU 检测 + vLLM 决策
# ============================================================================

def detect_gpus():
    """自动检测 GPU 数量"""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    except ImportError:
        return 0


def check_vllm_available():
    """检查 vLLM 是否可用"""
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def recommend_tp(n_gpus):
    """
    推荐张量并行数（TP）
        1 卡：TP=1
        2-4 卡：TP=2（收益最大）
        4+ 卡：TP=min(2, n_gpus)
    """
    if n_gpus <= 1:
        return 1
    elif n_gpus <= 4:
        return 2
    else:
        return min(2, n_gpus)


def run_subcommand(cmd_args: list, description: str = ""):
    """运行子命令（通过调用旧脚本实现，保持向后兼容）"""
    script_map = {
        "run": "eval_edu.py",
        "compare": "compare_evals.py",
        "errors": "analyze_errors.py",
        "meta": "meta_evaluation.py",
        "report": "generate_report.py",
    }
    sub = cmd_args[0]
    if sub not in script_map:
        print(f"未知子命令: {sub}")
        sys.exit(1)

    script = script_map[sub]
    cmd = [sys.executable, str(SCRIPT_DIR / script)] + cmd_args[1:]
    print(f"\n{'=' * 70}")
    print(f"调用: {description or sub}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")
    result = subprocess.run(cmd, env=os.environ.copy())
    return result.returncode == 0


def print_vllm_info(use_vllm, vllm_available, n_gpus, tp):
    """打印 vLLM 决策信息"""
    status = ""
    if not use_vllm:
        status = "用户禁用"
    elif not vllm_available:
        status = "vLLM 未安装 -> 降级到 HF"
    else:
        status = f"已启用（TP={tp}）"

    print(f"\n{'=' * 70}")
    print(f"vLLM 推理决策")
    print(f"{'=' * 70}")
    print(f"  检测到 {n_gpus} 张 GPU")
    print(f"  vLLM 安装: {'是' if vllm_available else '否'}")
    print(f"  -> vLLM 推理: {status}")
    print(f"{'=' * 70}\n")


def cmd_run(args):
    """运行评估（自动启用 vLLM）"""
    # 自动决策
    n_gpus = detect_gpus()
    vllm_available = check_vllm_available()
    use_vllm = getattr(args, "use_vllm", True)
    tp = args.tensor_parallel_size if args.tensor_parallel_size > 0 else recommend_tp(n_gpus)

    print_vllm_info(use_vllm, vllm_available, n_gpus, tp)

    cmd = ["run"]
    cmd.extend(["--stage", args.stage])
    if args.model_path:
        cmd.extend(["--model_path", args.model_path])
    if args.eval_all:
        cmd.append("--eval_all")
    cmd.extend(["--max_samples", str(args.max_samples)])
    if args.eval_data:
        cmd.extend(["--eval_data", args.eval_data])
    if args.save_raw_samples:
        cmd.append("--save_raw_samples")
    # 🆕 vLLM 参数
    if use_vllm and vllm_available:
        cmd.append("--use_vllm")
        if tp > 1:
            cmd.extend(["--tensor_parallel_size", str(tp)])
    return 0 if run_subcommand(cmd, f"运行 {args.stage} 阶段评估") else 1


def cmd_compare(args):
    """对比两次评估"""
    cmd = ["compare"]
    if args.baseline:
        cmd.extend(["--baseline", args.baseline])
    if args.final:
        cmd.extend(["--final", args.final])
    if args.show_weak:
        cmd.append("--show_weak")
    return 0 if run_subcommand(cmd, "评估对比") else 1


def cmd_errors(args):
    """错误案例分析"""
    cmd = ["errors"]
    if args.eval_file:
        cmd.extend(["--eval_file", args.eval_file])
    if args.output_errors:
        cmd.extend(["--output_errors", args.output_errors])
    if args.max_samples:
        cmd.extend(["--max_samples", str(args.max_samples)])
    return 0 if run_subcommand(cmd, "错误案例分析") else 1


def cmd_meta(args):
    """元评估"""
    cmd = ["meta"]
    if args.check_consistency:
        cmd.append("--check_consistency")
    if args.llm_judge:
        cmd.extend(["--llm_judge", args.llm_judge])
    if args.samples:
        cmd.extend(["--samples", str(args.samples)])
    return 0 if run_subcommand(cmd, "元评估") else 1


def cmd_report(args):
    """生成评估报告（🆕 自动带时间戳和效果命名）"""
    cmd = ["report"]
    if args.eval_file:
        cmd.extend(["--eval_file", args.eval_file])
    if args.baseline:
        cmd.extend(["--baseline", args.baseline])
    if args.final:
        cmd.extend(["--final", args.final])
    # 不传 --output，让 generate_report.py 自动生成带时间戳和效果的文件名
    if args.output:
        cmd.extend(["--output", args.output])
    return 0 if run_subcommand(cmd, "生成评估报告") else 1


def cmd_all(args):
    """一体化：run + meta + report（自动启用 vLLM）"""
    print("\n" + "=" * 70)
    print("一体化评估：run + meta + report（自动 vLLM）")
    print("=" * 70)

    # 步骤 1: 运行评估
    print(f"\n步骤 1/3: 运行 {args.stage} 评估")
    ret = cmd_run(args)
    if ret != 0:
        print("评估失败，终止")
        return 1

    # 步骤 2: 元评估
    print(f"\n步骤 2/3: 元评估")
    eval_file = args.eval_file or f"eval_results/{args.stage}.json"
    ret = cmd_meta(argparse.Namespace(
        eval_file=eval_file,
        check_consistency=True,
        llm_judge=None,
        samples=50,
    ))
    if ret != 0:
        print("元评估失败（继续）")

    # 步骤 3: 生成报告
    print(f"\n步骤 3/3: 生成报告")
    # 不传 output 时，generate_report.py 会自动生成带时间戳和效果的文件名
    ret = cmd_report(argparse.Namespace(
        eval_file=eval_file,
        baseline=None,
        final=None,
        output=args.report_output,
    ))
    if ret != 0:
        print("报告生成失败（继续）")

    print("\n" + "=" * 70)
    print("一体化评估完成")
    print(f"   评估结果: {eval_file}")
    if args.report_output:
        print(f"   评估报告: {args.report_output}")
    else:
        # 提示用户去 eval_results/ 目录查看
        from datetime import datetime
        today = datetime.now().strftime("%Y%m%d")
        print(f"   评估报告: eval_results/{args.stage}_{today}_*.md  (查看最新)")
    print("=" * 70)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="QwenSearch 一站式评估脚本（自动 vLLM）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s run --stage sft --model_path out/edu_sft --eval_all
  %(prog)s run --stage sft --model_path out/edu_sft --no_vllm
  %(prog)s all --stage sft --model_path out/edu_sft --eval_all
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # run 子命令
    p_run = subparsers.add_parser("run", help="运行评估（自动 vLLM）")
    p_run.add_argument("--stage", type=str, default="sft",
                       choices=["baseline", "sft", "grpo", "full", "fine"],
                       help="评估阶段")
    p_run.add_argument("--model_path", type=str, default=None, help="模型路径")
    p_run.add_argument("--eval_all", action="store_true", help="全量评估 19 个数据集")
    p_run.add_argument("--max_samples", type=int, default=200, help="每数据集最大样本数")
    p_run.add_argument("--eval_data", type=str, default=None, help="自定义评估数据")
    p_run.add_argument("--save_raw_samples", action="store_true", help="保存原始样本")
    # 🆕 vLLM 参数
    p_run.add_argument("--no_vllm", dest="use_vllm", action="store_false",
                       help="禁用 vLLM，使用 HF transformers")
    p_run.add_argument("--tensor_parallel_size", type=int, default=0,
                       help="张量并行数（0=自动）")
    p_run.add_argument("--eval_file", type=str, default=None, help="评估结果文件")
    p_run.set_defaults(func=cmd_run, use_vllm=True)

    # compare 子命令
    p_cmp = subparsers.add_parser("compare", help="对比两次评估")
    p_cmp.add_argument("--baseline", type=str, default=None, help="基线文件")
    p_cmp.add_argument("--final", type=str, default=None, help="最终文件")
    p_cmp.add_argument("--show_weak", action="store_true", help="显示弱项")
    p_cmp.set_defaults(func=cmd_compare)

    # errors 子命令
    p_err = subparsers.add_parser("errors", help="错误案例分析")
    p_err.add_argument("--eval_file", type=str, default=None, help="评估结果文件")
    p_err.add_argument("--output_errors", type=str, default=None, help="错误输出文件")
    p_err.add_argument("--max_samples", type=int, default=None, help="最大样本数")
    p_err.set_defaults(func=cmd_errors)

    # meta 子命令
    p_meta = subparsers.add_parser("meta", help="元评估")
    p_meta.add_argument("--check_consistency", action="store_true", help="检查一致性")
    p_meta.add_argument("--llm_judge", type=str, default=None, help="LLM 评判模型")
    p_meta.add_argument("--samples", type=int, default=50, help="样本数")
    p_meta.set_defaults(func=cmd_meta)

    # report 子命令
    p_rpt = subparsers.add_parser("report", help="生成评估报告")
    p_rpt.add_argument("--eval_file", type=str, default=None, help="评估结果文件")
    p_rpt.add_argument("--baseline", type=str, default=None, help="基线文件")
    p_rpt.add_argument("--final", type=str, default=None, help="最终文件")
    p_rpt.add_argument("--output", type=str, default=None, help="报告输出路径")
    p_rpt.set_defaults(func=cmd_report)

    # all 子命令
    p_all = subparsers.add_parser("all", help="一体化：run + meta + report（自动 vLLM）")
    p_all.add_argument("--stage", type=str, default="sft",
                       choices=["baseline", "sft", "grpo", "full", "fine"],
                       help="评估阶段")
    p_all.add_argument("--model_path", type=str, default=None, help="模型路径")
    p_all.add_argument("--eval_all", action="store_true", help="全量评估 19 个数据集")
    p_all.add_argument("--max_samples", type=int, default=200, help="每数据集最大样本数")
    p_all.add_argument("--eval_data", type=str, default=None, help="自定义评估数据")
    p_all.add_argument("--save_raw_samples", action="store_true", help="保存原始样本")
    p_all.add_argument("--no_vllm", dest="use_vllm", action="store_false",
                       help="禁用 vLLM")
    p_all.add_argument("--tensor_parallel_size", type=int, default=0, help="张量并行数（0=自动）")
    p_all.add_argument("--eval_file", type=str, default=None, help="评估结果文件")
    p_all.add_argument("--report_output", type=str, default=None, help="报告输出路径")
    p_all.set_defaults(func=cmd_all, use_vllm=True)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
