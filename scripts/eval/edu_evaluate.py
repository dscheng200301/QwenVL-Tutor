"""
QwenSearch 一站式评估脚本

合并功能（替�?8 个分散脚本）:
    run       - 运行评估（替�?eval_edu.py�?    compare   - 对比两次评估（替�?compare_evals.py�?    errors    - 错误案例分析（替�?analyze_errors.py�?    meta      - 元评估（替代 meta_evaluation.py�?    report    - 生成评估报告（替�?generate_report.py�?    all       - 一体化：run + meta + report

向后兼容:
    旧脚本（eval_edu.py, compare_evals.py 等）仍可独立运行
    本脚本通过子命令提供统一入口

用法:
    # 基础评估
    python scripts/edu_evaluate.py run --stage sft --model_path out/edu_sft --eval_all

    # 训练前基�?    python scripts/edu_evaluate.py run --stage baseline --eval_all --max_samples 200

    # 对比两次评估
    python scripts/edu_evaluate.py compare --show_weak

    # 错误归类
    python scripts/edu_evaluate.py errors --output_errors errors.json

    # 元评估（指标一致性）
    python scripts/edu_evaluate.py meta --check_consistency

    # 生成报告
    python scripts/edu_evaluate.py report --output report.md

    # 一体化（推荐：训练后一次跑完）
    python scripts/edu_evaluate.py all --stage sft --model_path out/edu_sft --eval_all
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR.parent.parent)


def run_subcommand(cmd_args: list, description: str = ""):
    """运行子命令（通过调用旧脚本实现，保持向后兼容�?""
    script_map = {
        "run": "eval_edu.py",
        "compare": "compare_evals.py",
        "errors": "analyze_errors.py",
        "meta": "meta_evaluation.py",
        "report": "generate_report.py",
    }
    sub = cmd_args[0]
    if sub not in script_map:
        print(f"�?未知子命�? {sub}")
        sys.exit(1)

    script = script_map[sub]
    # args[0] 是子命令名，需要去�?    cmd = [sys.executable, str(SCRIPT_DIR / script)] + cmd_args[1:]
    print(f"\n{'=' * 70}")
    print(f"🔧 调用: {description or sub}")
    print(f"   命令: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")
    result = subprocess.run(cmd, env=os.environ.copy())
    return result.returncode == 0


def cmd_run(args):
    """运行评估"""
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
    return 0 if run_subcommand(cmd, f"运行 {args.stage} 阶段评估") else 1


def cmd_compare(args):
    """对比两次评估"""
    cmd = ["compare"]
    if args.file1:
        cmd.extend(["--file1", args.file1])
    if args.file2:
        cmd.extend(["--file2", args.file2])
    if args.show_weak:
        cmd.append("--show_weak_datasets")
    return 0 if run_subcommand(cmd, "对比两次评估结果") else 1


def cmd_errors(args):
    """错误案例分析"""
    cmd = ["errors"]
    if args.eval_file:
        cmd.extend(["--eval_file", args.eval_file])
    if args.output_errors:
        cmd.extend(["--output_errors", args.output_errors])
    return 0 if run_subcommand(cmd, "错误案例分析") else 1


def cmd_meta(args):
    """元评�?""
    cmd = ["meta"]
    if args.check_consistency:
        cmd.append("--check_consistency")
    if args.llm_judge:
        cmd.extend(["--llm_judge", args.llm_judge])
        cmd.extend(["--samples", str(args.samples)])
    if args.audit:
        cmd.append("--audit")
    if args.monthly_report:
        cmd.append("--monthly_report")
        if args.output:
            cmd.extend(["--output", args.output])
    return 0 if run_subcommand(cmd, "元评�?) else 1


def cmd_report(args):
    """生成评估报告"""
    cmd = ["report"]
    if args.eval_files:
        cmd.extend(["--eval_files"] + args.eval_files)
    if args.mode:
        cmd.extend(["--mode", args.mode])
    if args.baseline:
        cmd.extend(["--baseline", args.baseline])
    if args.final:
        cmd.extend(["--final", args.final])
    if args.output:
        cmd.extend(["--output", args.output])
    return 0 if run_subcommand(cmd, "生成评估报告") else 1


def cmd_all(args):
    """一体化：run + meta + report"""
    print("\n" + "=" * 70)
    print("🚀 一体化评估：run �?meta �?report")
    print("=" * 70)

    steps = [
        ("run", f"步骤 1/3: 运行 {args.stage} 阶段评估"),
        ("meta", "步骤 2/3: 元评估（指标一致性检查）"),
        ("report", "步骤 3/3: 生成评估报告"),
    ]

    for sub, desc in steps:
        print(f"\n�?{desc}")
        if sub == "run":
            ret = cmd_run(args)
        elif sub == "meta":
            ret = cmd_meta(argparse.Namespace(
                check_consistency=True, llm_judge=None, samples=50,
                audit=False, monthly_report=False, output=None,
            ))
        elif sub == "report":
            ret = cmd_report(argparse.Namespace(
                eval_files=None, mode=None, baseline=None, final=None,
                output=args.output or "report.md",
            ))
        if ret != 0:
            print(f"\n⚠️ {sub} 步骤失败，但继续执行下一�?)
    print("\n" + "=" * 70)
    print("�?一体化评估完成�?)
    print("=" * 70)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="QwenSearch 一站式评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s run --stage sft --model_path out/edu_sft --eval_all
  %(prog)s compare --show_weak
  %(prog)s errors --output_errors errors.json
  %(prog)s meta --check_consistency
  %(prog)s report --output report.md
  %(prog)s all --stage sft --model_path out/edu_sft --eval_all
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命�?)

    # run 子命�?    p_run = subparsers.add_parser("run", help="运行评估")
    p_run.add_argument("--stage", type=str, default="sft",
                        choices=["baseline", "sft", "grpo", "full", "fine"],
                        help="评估阶段")
    p_run.add_argument("--model_path", type=str, default=None, help="模型路径")
    p_run.add_argument("--eval_all", action="store_true", help="评估所�?19 个数据集")
    p_run.add_argument("--max_samples", type=int, default=200, help="每数据集最大样本数")
    p_run.add_argument("--eval_data", type=str, default=None, help="自定义评估数据路�?)
    p_run.add_argument("--save_raw_samples", action="store_true",
                        help="保存原始样本（用于错误分析）")
    p_run.set_defaults(func=cmd_run)

    # compare 子命�?    p_cmp = subparsers.add_parser("compare", help="对比两次评估")
    p_cmp.add_argument("--file1", type=str, default=None, help="第一个评估文�?)
    p_cmp.add_argument("--file2", type=str, default=None, help="第二个评估文�?)
    p_cmp.add_argument("--show_weak", action="store_true", help="显示弱项数据�?)
    p_cmp.set_defaults(func=cmd_compare)

    # errors 子命�?    p_err = subparsers.add_parser("errors", help="错误案例分析")
    p_err.add_argument("--eval_file", type=str, default=None, help="评估结果文件")
    p_err.add_argument("--output_errors", type=str, default=None, help="导出错误样本")
    p_err.set_defaults(func=cmd_errors)

    # meta 子命�?    p_meta = subparsers.add_parser("meta", help="元评�?)
    p_meta.add_argument("--check_consistency", action="store_true", help="检查指标一致�?)
    p_meta.add_argument("--llm_judge", type=str, default=None, help="LLM Judge 模型（如 gpt-4o�?)
    p_meta.add_argument("--samples", type=int, default=50, help="LLM Judge 样本�?)
    p_meta.add_argument("--audit", action="store_true", help="生成人工抽查样本")
    p_meta.add_argument("--monthly_report", action="store_true", help="生成月度报告")
    p_meta.add_argument("--output", type=str, default=None, help="输出文件")
    p_meta.set_defaults(func=cmd_meta)

    # report 子命�?    p_rep = subparsers.add_parser("report", help="生成评估报告")
    p_rep.add_argument("--eval_files", nargs="+", default=None, help="评估文件列表")
    p_rep.add_argument("--mode", type=str, default=None,
                        choices=["single", "compare"], help="报告模式")
    p_rep.add_argument("--baseline", type=str, default=None, help="对比模式基线")
    p_rep.add_argument("--final", type=str, default=None, help="对比模式最�?)
    p_rep.add_argument("--output", type=str, default=None, help="报告输出路径")
    p_rep.set_defaults(func=cmd_report)

    # all 子命令（一站式�?    p_all = subparsers.add_parser("all", help="一体化：run + meta + report")
    p_all.add_argument("--stage", type=str, default="sft",
                        choices=["baseline", "sft", "grpo", "full"],
                        help="评估阶段")
    p_all.add_argument("--model_path", type=str, default=None, help="模型路径")
    p_all.add_argument("--eval_all", action="store_true", help="评估所有数据集")
    p_all.add_argument("--max_samples", type=int, default=200, help="每数据集最大样本数")
    p_all.add_argument("--eval_data", type=str, default=None, help="自定义评估数据路�?)
    p_all.add_argument("--output", type=str, default="report.md", help="报告输出路径")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
