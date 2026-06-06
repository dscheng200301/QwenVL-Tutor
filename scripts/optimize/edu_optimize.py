"""
QwenVL-Tutor All-in-One Optimization Script

Combined functions (replaces 3 separate scripts):
    resample  - Data resampling (replaces resample_data.py)
    build     - GRPO training data preparation
    retrain   - Trigger retraining (wraps train_sft.py)
    auto      - Integrated: auto resample + retrain based on eval results
    grpo      - GRPO post-eval optimization loop (auto-decide: retry / fallback SFT)

Backward compatibility:
    Old scripts (resample_data.py) can still run independently
    This script provides a unified entry point

Usage:
    python scripts/edu_optimize.py resample --output weights.json
    python scripts/edu_optimize.py build --output edu_grpo.parquet
    python scripts/edu_optimize.py retrain --data_paths "..." --epochs 2
    python scripts/edu_optimize.py auto --epochs 2
    python scripts/edu_optimize.py grpo --eval_file eval_results/grpo_xxx.json
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR.parent.parent)


def run_subcommand(cmd: list, description: str = ""):
    """Run subcommand via subprocess"""
    print(f"\n{'=' * 70}")
    print(f"  Calling: {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")
    result = subprocess.run(cmd, env=os.environ.copy())
    return result.returncode == 0


# ============================================================================
# Subcommand implementations
# ============================================================================


def cmd_resample(args):
    """Data resample v2 (wraps resample_data.py)"""
    cmd = [sys.executable, str(SCRIPT_DIR / "resample_data.py")]

    if args.eval_file:
        cmd.extend(["--eval_file", args.eval_file])
    if args.output:
        cmd.extend(["--output", args.output])
    if args.v1:
        cmd.append("--v1")
    cmd.extend(["--min_weight", str(args.min_weight)])
    cmd.extend(["--max_weight", str(args.max_weight)])

    return 0 if run_subcommand(cmd, "Data resample v2") else 1


def cmd_build(args):
    """Generate GRPO training data (5K curated)"""

    print("\n" + "=" * 70)
    print("  Building GRPO data")
    print("=" * 70)

    import pyarrow as pa
    import pyarrow.parquet as pq
    import random

    GRPO_DATASETS = {
        'cmmu': ('dataset/edu_cmmu.parquet', 1500, 'Chinese K12 7 core subjects'),
        'mmscibench': ('dataset/edu_mmscibench.parquet', 1000, 'Middle school math+physics, detailed reasoning'),
        'math_verse': ('dataset/edu_math_verse.parquet', 1000, 'Math visual reasoning'),
        'math_vista': ('dataset/edu_math_vista.parquet', 500, 'Visual reasoning benchmark'),
        'scienceqa': ('dataset/edu_science.parquet', 1000, 'Complete reasoning chain'),
    }

    random.seed(42)
    all_tables = []
    total_target = 0
    for name, (path, max_samples, note) in GRPO_DATASETS.items():
        if not os.path.exists(path):
            print(f"  WARNING: dataset {name} not found: {path}")
            continue
        print(f"  {name} (target {max_samples}): {note}")
        try:
            table = pq.read_table(path)
            before = len(table)
            if len(table) > max_samples:
                indices = sorted(random.sample(range(len(table)), max_samples))
                table = table.take(indices)
            all_tables.append(table)
            total_target += max_samples
            print(f"   original {before} -> sampled {len(table)}\n")
        except Exception as e:
            print(f"   ERROR loading: {e}\n")

    if not all_tables:
        print("ERROR: No usable data. Please run: python scripts/download_all_data.py")
        return 1

    merged = pa.concat_tables(all_tables)
    output = args.output or 'dataset/edu_grpo.parquet'
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    pq.write_table(merged, output)
    print("=" * 70)
    print(f"SUCCESS: GRPO data ready: {len(merged)} rows (target {total_target})")
    print(f"  Saved: {output}")
    print("=" * 70)
    return 0


def cmd_retrain(args):
    """Trigger retraining (wraps train_sft.py)"""
    cmd = [sys.executable, "trainer/train_sft.py"]

    if args.data_paths:
        cmd.extend(["--data_paths", args.data_paths])
    if args.from_weight:
        cmd.extend(["--from_weight", args.from_weight])
    cmd.extend(["--epochs", str(args.epochs)])
    if args.save_weight:
        cmd.extend(["--save_weight", args.save_weight])
    if args.batch_size:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.learning_rate:
        cmd.extend(["--learning_rate", str(args.learning_rate)])

    return 0 if run_subcommand(cmd, f"Retrain {args.epochs} epochs") else 1


def cmd_auto(args):
    """Integrated: auto resample + retrain based on eval results"""

    print("\n" + "=" * 70)
    print("  Integrated optimization: resample -> read weights -> retrain")
    print("=" * 70)

    # Step 1: resample
    print("\n--- Step 1/3: Resample")
    weights_path = args.weights or "weights.json"
    ret = cmd_resample(argparse.Namespace(
        eval_file=args.eval_file, output=weights_path, v1=False,
        min_weight=0.5, max_weight=2.5,
    ))
    if ret != 0:
        print("ERROR: Resample failed, aborting")
        return 1

    # Step 2: read weights and build data_paths
    print("\n--- Step 2/3: Read resample weights")
    if not os.path.exists(weights_path):
        print(f"ERROR: Weight file not found: {weights_path}")
        return 1
    with open(weights_path, 'r', encoding='utf-8') as f:
        weights_data = json.load(f)
    data_paths = weights_data.get("data_paths", "")
    if not data_paths:
        print("WARNING: No data_paths in weight file, using all datasets")
        data_paths = ""
    print(f"   Loaded {len(weights_data.get('weights', {}))} dataset weights")
    print(f"   data_paths length: {len(data_paths)} chars")

    # Step 3: trigger retrain
    print(f"\n--- Step 3/3: Trigger retrain ({args.epochs} epochs)")
    save_weight = args.save_weight or f"edu_sft_v{int(__import__('time').time())}"
    ret = cmd_retrain(argparse.Namespace(
        data_paths=data_paths, from_weight=args.from_weight,
        epochs=args.epochs, save_weight=save_weight,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
    ))
    if ret != 0:
        print("ERROR: Retrain failed")
        return 1

    print("\n" + "=" * 70)
    print("SUCCESS: Integrated optimization complete!")
    print(f"   Weight file: {weights_path}")
    print(f"   Saved model: out/{save_weight}")
    print("=" * 70)
    return 0


def cmd_grpo(args):
    """GRPO post-eval optimization: auto-decide based on eval results"""
    print("\n" + "=" * 70)
    print("GRPO Optimization Loop: evaluate -> decide -> execute")
    print("=" * 70)

    # Step 1: load GRPO eval results
    eval_file = args.eval_file
    if not eval_file:
        eval_file = _find_latest_eval("grpo")
        if not eval_file:
            print("No GRPO eval results found. Run eval first or specify --eval_file")
            return 1
    print(f"\n[1/4] Load GRPO eval: {eval_file}")
    with open(eval_file, 'r', encoding='utf-8') as f:
        grpo_results = json.load(f)
    grpo_acc = _get_avg_accuracy(grpo_results)
    print(f"  GRPO weighted accuracy: {grpo_acc:.4f}")

    # Step 2: find SFT eval for comparison
    sft_file = args.sft_eval_file
    if not sft_file:
        sft_file = _find_latest_eval("sft")
    sft_acc = None
    if sft_file and os.path.exists(sft_file):
        with open(sft_file, 'r', encoding='utf-8') as f:
            sft_results = json.load(f)
        sft_acc = _get_avg_accuracy(sft_results)
        print(f"\n[2/4] Compare with SFT eval: {sft_file}")
        print(f"  SFT weighted accuracy: {sft_acc:.4f}")
        delta = grpo_acc - sft_acc
        if sft_acc > 0:
            print(f"  GRPO vs SFT: {delta:+.4f} ({delta / sft_acc * 100:+.1f}%)")
    else:
        print(f"\n[2/4] No SFT eval for comparison, checking GRPO quality only")

    # Step 3: decide
    print(f"\n[3/4] Decision")

    if sft_acc is not None and grpo_acc < sft_acc * 0.95:
        print(f"  Verdict: Severe degradation (GRPO {grpo_acc:.4f} < SFT {sft_acc:.4f} * 0.95)")
        print(f"  Action: Adjust GRPO hyperparams and retry")
        ret = _grpo_retry_adjusted(args, grpo_results)
        if ret == 0:
            print("\n" + "=" * 70)
            print("GRPO optimization done (hyperparam-adjusted retry)")
            print("=" * 70)
        return ret

    if sft_acc is not None and grpo_acc < sft_acc * 1.02:
        reason = "slight degradation" if grpo_acc < sft_acc else "insufficient improvement"
        print(f"  Verdict: {reason} (GRPO {grpo_acc:.4f} vs SFT {sft_acc:.4f})")
        print(f"  Action: Fall back to SFT optimization loop (resample -> retrain SFT -> re-GRPO)")
        ret = _grpo_fallback_sft(args, grpo_results)
        if ret == 0:
            print("\n" + "=" * 70)
            print("GRPO optimization done (SFT fallback loop)")
            print("Next: python trainer/train_grpo.py --from_weight ../out/<new_sft_weight>")
            print("=" * 70)
        return ret

    if sft_acc is not None and grpo_acc < sft_acc * 1.05:
        delta = (grpo_acc - sft_acc) / sft_acc * 100
        print(f"  Verdict: Improved but not ideal (+{delta:.1f}%)")
        print(f"  Action: SFT optimization + GRPO hyperparam tuning")
        return _grpo_combined_optimize(args, grpo_results)

    if sft_acc is not None:
        improvement = (grpo_acc - sft_acc) / sft_acc * 100
    else:
        improvement = 0
    print(f"  Verdict: GRPO improvement is good (+{improvement:.1f}%)")
    print(f"  Action: No optimization needed, proceed to final evaluation")
    return 0


def _get_avg_accuracy(results: dict) -> float:
    """Extract weighted accuracy from eval results"""
    if "aggregate" in results and "weighted_accuracy" in results["aggregate"]:
        return results["aggregate"]["weighted_accuracy"]
    if "datasets" in results:
        accs = [v.get("accuracy", 0) for v in results["datasets"].values()]
        return sum(accs) / max(len(accs), 1)
    return 0.0


def _find_latest_eval(stage: str):
    """Find the latest eval result file for a given stage"""
    import glob as _glob
    pattern = f"eval_results/{stage}_*.json"
    files = sorted(_glob.glob(pattern), reverse=True)
    return files[0] if files else None


def _grpo_retry_adjusted(args, grpo_results: dict) -> int:
    """Retry GRPO with adjusted hyperparams"""
    print(f"\n  [4/4] Retry GRPO with adjusted hyperparams")
    from_weight = args.from_weight or grpo_results.get("model_path", "../out/edu_sft")
    save_weight = args.save_weight or f"edu_grpo_v{int(__import__('time').time()) % 100000}"

    cmd = [
        sys.executable, "trainer/train_grpo.py",
        "--from_weight", from_weight,
        "--epochs", str(args.epochs or 1),
        "--num_generations", str(args.num_generations or 6),
        "--learning_rate", str(args.learning_rate or 5e-8),
        "--save_weight", save_weight,
    ]
    if args.api_model:
        cmd += ["--api_model", args.api_model]
    if args.api_key:
        cmd += ["--api_key", args.api_key]
    if args.data_path:
        cmd += ["--data_path", args.data_path]

    print(f"  Adjusted: K={args.num_generations or 6}, lr={args.learning_rate or 5e-8}")
    return 0 if run_subcommand(cmd, f"GRPO retry ({save_weight})") else 1


def _grpo_fallback_sft(args, grpo_results: dict) -> int:
    """Fall back to SFT optimization loop"""
    print(f"\n  [4/4] Fall back: SFT resample -> retrain SFT")
    sft_file = args.sft_eval_file or _find_latest_eval("sft")

    # resample
    weights_path = args.weights or "weights.json"
    ret = cmd_resample(argparse.Namespace(
        eval_file=sft_file, output=weights_path, v1=False,
        min_weight=0.5, max_weight=2.5,
    ))
    if ret != 0:
        return 1

    # retrain SFT
    if not os.path.exists(weights_path):
        return 1
    with open(weights_path, 'r', encoding='utf-8') as f:
        weights_data = json.load(f)
    data_paths = weights_data.get("data_paths", "")

    save_weight = args.save_weight or f"edu_sft_v{int(__import__('time').time()) % 100000}"
    return cmd_retrain(argparse.Namespace(
        data_paths=data_paths, from_weight=None,
        epochs=args.epochs or 2, save_weight=save_weight,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
    ))


def _grpo_combined_optimize(args, grpo_results: dict) -> int:
    """Combined: SFT optimization + GRPO hyperparam tuning"""
    print(f"\n  [4/4] Combined: SFT optimization -> GRPO hyperparam tuning")

    ret = _grpo_fallback_sft(args, grpo_results)
    if ret != 0:
        return ret

    new_args = argparse.Namespace(
        from_weight=grpo_results.get("model_path", "../out/edu_sft"),
        save_weight=args.save_weight or f"edu_grpo_v{int(__import__('time').time()) % 100000}",
        epochs=args.epochs or 1,
        num_generations=args.num_generations or 6,
        learning_rate=args.learning_rate or 5e-8,
        api_model=args.api_model,
        api_key=args.api_key,
        data_path=args.data_path,
        eval_file=args.eval_file,
        sft_eval_file=_find_latest_eval("sft"),
        weights=args.weights,
        batch_size=args.batch_size,
    )
    return _grpo_retry_adjusted(new_args, grpo_results)


def main():
    parser = argparse.ArgumentParser(
        description="QwenVL-Tutor All-in-One Optimization Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s resample --output weights.json
  %(prog)s build --output edu_grpo.parquet
  %(prog)s retrain --data_paths "..." --epochs 2
  %(prog)s auto --epochs 2 --save_weight edu_sft_v2
  %(prog)s grpo --eval_file eval_results/grpo_xxx.json
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # resample subcommand
    p_res = subparsers.add_parser("resample", help="Data resampling v2")
    p_res.add_argument("--eval_file", type=str, default=None, help="Eval result file")
    p_res.add_argument("--output", type=str, default=None, help="Weight output JSON")
    p_res.add_argument("--v1", action="store_true", help="Use v1 old formula")
    p_res.add_argument("--min_weight", type=float, default=0.5, help="Weight lower bound")
    p_res.add_argument("--max_weight", type=float, default=2.5, help="Weight upper bound")
    p_res.set_defaults(func=cmd_resample)

    # build subcommand
    p_bld = subparsers.add_parser("build", help="Generate GRPO training data")
    p_bld.add_argument("--output", type=str, default=None, help="Output parquet path")
    p_bld.set_defaults(func=cmd_build)

    # retrain subcommand
    p_ret = subparsers.add_parser("retrain", help="Trigger retraining")
    p_ret.add_argument("--data_paths", type=str, default=None, help="Training data paths (comma separated)")
    p_ret.add_argument("--from_weight", type=str, default=None, help="Starting weight")
    p_ret.add_argument("--epochs", type=int, default=2, help="Training epochs")
    p_ret.add_argument("--save_weight", type=str, default=None, help="Save weight name")
    p_ret.add_argument("--batch_size", type=int, default=None, help="Batch size")
    p_ret.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    p_ret.set_defaults(func=cmd_retrain)

    # auto subcommand (all-in-one)
    p_auto = subparsers.add_parser("auto", help="Integrated: resample + retrain")
    p_auto.add_argument("--eval_file", type=str, default=None, help="Eval result file")
    p_auto.add_argument("--weights", type=str, default="weights.json", help="Weight output")
    p_auto.add_argument("--from_weight", type=str, default=None, help="Starting weight")
    p_auto.add_argument("--epochs", type=int, default=2, help="Training epochs")
    p_auto.add_argument("--save_weight", type=str, default=None, help="Save weight name")
    p_auto.add_argument("--batch_size", type=int, default=None, help="Batch size")
    p_auto.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    p_auto.set_defaults(func=cmd_auto)

    # grpo subcommand: GRPO post-eval optimization loop
    p_grpo = subparsers.add_parser("grpo", help="GRPO post-eval optimization (auto-decide: retry / fallback SFT)")
    p_grpo.add_argument("--eval_file", type=str, default=None, help="GRPO eval result file")
    p_grpo.add_argument("--sft_eval_file", type=str, default=None, help="SFT eval result file for comparison")
    p_grpo.add_argument("--from_weight", type=str, default=None, help="GRPO source weight path")
    p_grpo.add_argument("--save_weight", type=str, default=None, help="Save weight name")
    p_grpo.add_argument("--epochs", type=int, default=None, help="Training epochs")
    p_grpo.add_argument("--num_generations", type=int, default=None, help="GRPO K candidates (default 6)")
    p_grpo.add_argument("--learning_rate", type=float, default=None, help="Learning rate (default 5e-8)")
    p_grpo.add_argument("--api_model", type=str, default=None, help="LLM judge model")
    p_grpo.add_argument("--api_key", type=str, default=None, help="API key")
    p_grpo.add_argument("--data_path", type=str, default=None, help="GRPO training data path")
    p_grpo.add_argument("--weights", type=str, default="weights.json", help="Resample weights output")
    p_grpo.add_argument("--batch_size", type=int, default=None, help="Batch size")
    p_grpo.set_defaults(func=cmd_grpo)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())