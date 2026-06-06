"""
QwenVL-Tutor 一站式优化脚本

合并功能（替代 3 个分散脚本）:
    resample  - 数据重采样（替代 resample_data.py）
    build     - GRPO 强化数据准备
    retrain   - 触发再训练（封装 train_sft.py）
    auto      - 一体化：基于评估结果自动 resample + retrain

向后兼容:
    旧脚本（resample_data.py）仍可独立运行
    本脚本提供统一入口

用法:
    # 1. åºäºè¯ä¼°ç»æééæ ?    python scripts/edu_optimize.py resample --output weights.json

    # 2. çæ GRPO å¼ºåæ°æ®
    python scripts/edu_optimize.py build --output edu_grpo.parquet

    # 3. è§¦ååè®­ç»?    python scripts/edu_optimize.py retrain --data_paths "dataset/edu_science.parquet,..." --epochs 2

    # 4. ä¸ä½åï¼èªå¨å³ç­?    #    æµç¨ï¼resample â?è¯»å weights â?retrain
    python scripts/edu_optimize.py auto --epochs 2
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
    """è¿è¡å­å½ä»¤ï¼éè¿ subprocess è°ç¨ï¼?""
    print(f"\n{'=' * 70}")
    print(f"ð§ è°ç¨: {description}")
    print(f"   å½ä»¤: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")
    result = subprocess.run(cmd, env=os.environ.copy())
    return result.returncode == 0


# ============================================================================
# å­å½ä»¤å®ç?# ============================================================================

def cmd_resample(args):
    """æ°æ®ééæ ?v2ï¼å°è£?resample_data.pyï¼?""
    cmd = [sys.executable, str(SCRIPT_DIR / "resample_data.py")]
    if args.eval_file:
        cmd.extend(["--eval_file", args.eval_file])
    if args.output:
        cmd.extend(["--output", args.output])
    if args.v1:
        cmd.append("--v1")
    cmd.extend(["--min_weight", str(args.min_weight)])
    cmd.extend(["--max_weight", str(args.max_weight)])
    return 0 if run_subcommand(cmd, "æ°æ®ééæ ?v2") else 1


def cmd_build(args):
    """çæ GRPO å¼ºåæ°æ®ï¼?K ç²¾éï¼"""
    print("\n" + "=" * 70)
    print("ð¨ çæ GRPO å¼ºåæ°æ®")
    print("=" * 70)

    import pyarrow as pa
    import pyarrow.parquet as pq
    import random

    GRPO_DATASETS = {
        'cmmu': ('dataset/edu_cmmu.parquet', 1500, 'ä¸­æ K12 7 é¨æ ¸å¿å­¦ç§?),
        'mmscibench': ('dataset/edu_mmscibench.parquet', 1000, 'ä¸­å­¦æ°å­¦+ç©çï¼éè¯¦ç»æ¨ç'),
        'math_verse': ('dataset/edu_math_verse.parquet', 1000, 'æ°å­¦è§è§æ¨ç'),
        'math_vista': ('dataset/edu_math_vista.parquet', 500, 'è§è§æ¨çåºå'),
        'scienceqa': ('dataset/edu_science.parquet', 1000, 'å®æ´æ¨çé?),
    }

    random.seed(42)
    all_tables = []
    total_target = 0
    for name, (path, max_samples, note) in GRPO_DATASETS.items():
        if not os.path.exists(path):
            print(f"  â ï¸ æ°æ®é?{name} ä¸å­å? {path}")
            continue
        print(f"ð¦ {name} (ç®æ  {max_samples}): {note}")
        try:
            table = pq.read_table(path)
            before = len(table)
            if len(table) > max_samples:
                indices = sorted(random.sample(range(len(table)), max_samples))
                table = table.take(indices)
            all_tables.append(table)
            total_target += max_samples
            print(f"   åå§ {before} -> éæ · {len(table)}\n")
        except Exception as e:
            print(f"   â?å è½½å¤±è´¥: {e}\n")

    if not all_tables:
        print("â?æ²¡æå¯ç¨æ°æ®ï¼è¯·åè¿è¡? python scripts/download_all_data.py")
        return 1

    merged = pa.concat_tables(all_tables)
    output = args.output or 'dataset/edu_grpo.parquet'
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    pq.write_table(merged, output)
    print("=" * 70)
    print(f"â?GRPO æ°æ®åå¤å®æ: {len(merged)} æ¡ï¼ç®æ  {total_target}ï¼?)
    print(f"ð¾ å·²ä¿å­? {output}")
    print("=" * 70)
    return 0


def cmd_retrain(args):
    """è§¦ååè®­ç»ï¼å°è£ train_sft.pyï¼?""
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
    return 0 if run_subcommand(cmd, f"åè®­ç»?{args.epochs} ä¸?epoch") else 1


def cmd_auto(args):
    """ä¸ä½åï¼åºäºè¯ä¼°ç»æèªå¨ééæ · + åè®­ç»?""
    print("\n" + "=" * 70)
    print("ð ä¸ä½åä¼åï¼resample â?è¯»å weights â?retrain")
    print("=" * 70)

    # æ­¥éª¤ 1: ééæ ?    print("\nâ?æ­¥éª¤ 1/3: ééæ ?)
    weights_path = args.weights or "weights.json"
    ret = cmd_resample(argparse.Namespace(
        eval_file=args.eval_file, output=weights_path, v1=False,
        min_weight=0.5, max_weight=2.5,
    ))
    if ret != 0:
        print("â?ééæ ·å¤±è´¥ï¼ç»æ­¢")
        return 1

    # æ­¥éª¤ 2: è¯»å weights å¹¶çæ?data_paths
    print("\nâ?æ­¥éª¤ 2/3: è¯»åééæ ·æé?)
    if not os.path.exists(weights_path):
        print(f"â?æéæä»¶ä¸å­å? {weights_path}")
        return 1
    with open(weights_path, 'r', encoding='utf-8') as f:
        weights_data = json.load(f)
    data_paths = weights_data.get("data_paths", "")
    if not data_paths:
        print("â ï¸ æéæä»¶ä¸­æ  data_pathsï¼ä½¿ç¨æææ°æ®é")
        data_paths = ""
    print(f"   å·²å è½?{len(weights_data.get('weights', {}))} ä¸ªæ°æ®éæé")
    print(f"   data_paths é¿åº¦: {len(data_paths)} å­ç¬¦")

    # æ­¥éª¤ 3: è§¦ååè®­ç»?    print(f"\nâ?æ­¥éª¤ 3/3: è§¦ååè®­ç»ï¼{args.epochs} epochsï¼?)
    save_weight = args.save_weight or f"edu_sft_v{int(__import__('time').time())}"
    ret = cmd_retrain(argparse.Namespace(
        data_paths=data_paths, from_weight=args.from_weight,
        epochs=args.epochs, save_weight=save_weight,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
    ))
    if ret != 0:
        print("â?åè®­ç»å¤±è´?)
        return 1

    print("\n" + "=" * 70)
    print("â?ä¸ä½åä¼åå®æï¼?)
    print(f"   æéæä»¶: {weights_path}")
    print(f"   ä¿å­æ¨¡å: out/{save_weight}")
    print("=" * 70)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="QwenVL-Tutor ä¸ç«å¼ä¼åèæ¬",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ç¤ºä¾:
  %(prog)s resample --output weights.json
  %(prog)s build --output edu_grpo.parquet
  %(prog)s retrain --data_paths "..." --epochs 2
  %(prog)s auto --epochs 2 --save_weight edu_sft_v2
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="å­å½ä»?)

    # resample å­å½ä»?    p_res = subparsers.add_parser("resample", help="æ°æ®ééæ ?v2")
    p_res.add_argument("--eval_file", type=str, default=None, help="è¯ä¼°ç»ææä»¶")
    p_res.add_argument("--output", type=str, default=None, help="æéè¾åº JSON")
    p_res.add_argument("--v1", action="store_true", help="ä½¿ç¨ v1 æ§å¬å¼?)
    p_res.add_argument("--min_weight", type=float, default=0.5, help="æéä¸é")
    p_res.add_argument("--max_weight", type=float, default=2.5, help="æéä¸é")
    p_res.set_defaults(func=cmd_resample)

    # build å­å½ä»?    p_bld = subparsers.add_parser("build", help="çæ GRPO å¼ºåæ°æ®")
    p_bld.add_argument("--output", type=str, default=None, help="è¾åº parquet è·¯å¾")
    p_bld.set_defaults(func=cmd_build)

    # retrain å­å½ä»?    p_ret = subparsers.add_parser("retrain", help="è§¦ååè®­ç»?)
    p_ret.add_argument("--data_paths", type=str, default=None, help="è®­ç»æ°æ®è·¯å¾ï¼éå·åéï¼?)
    p_ret.add_argument("--from_weight", type=str, default=None, help="èµ·å§æé")
    p_ret.add_argument("--epochs", type=int, default=2, help="è®­ç»è½®æ°")
    p_ret.add_argument("--save_weight", type=str, default=None, help="ä¿å­æéå?)
    p_ret.add_argument("--batch_size", type=int, default=None, help="æ¹å¤§å°?)
    p_ret.add_argument("--learning_rate", type=float, default=None, help="å­¦ä¹ ç?)
    p_ret.set_defaults(func=cmd_retrain)

    # auto å­å½ä»¤ï¼ä¸ç«å¼ï¼?    p_auto = subparsers.add_parser("auto", help="ä¸ä½åï¼resample + retrain")
    p_auto.add_argument("--eval_file", type=str, default=None, help="è¯ä¼°ç»ææä»¶")
    p_auto.add_argument("--weights", type=str, default="weights.json", help="æéè¾åº")
    p_auto.add_argument("--from_weight", type=str, default=None, help="èµ·å§æé")
    p_auto.add_argument("--epochs", type=int, default=2, help="è®­ç»è½®æ°")
    p_auto.add_argument("--save_weight", type=str, default=None, help="ä¿å­æéå?)
    p_auto.add_argument("--batch_size", type=int, default=None, help="æ¹å¤§å°?)
    p_auto.add_argument("--learning_rate", type=float, default=None, help="å­¦ä¹ ç?)
    p_auto.set_defaults(func=cmd_auto)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
