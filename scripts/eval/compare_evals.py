"""
QwenSearch è¯ä¼°ç»æå¯¹æ¯å·¥å·ï¼åçº§çï¼?
åè½:
    1. èªå¨æ¾å°ææ°çä¸¤ä¸ªè¯ä¼°æä»¶
    2. éææ å¯¹æ¯ï¼å?95% ç½®ä¿¡åºé´ï¼?    3. è®¡ç®ç»è®¡æ¾èæ§ï¼éå¯¹ Bootstrap æ£éªï¼
    4. è¾åºååè¶å¿å¤å®
    5. æ è®°å¼±é¡¹æ°æ®é?
ç¨æ³:
    # å¯¹æ¯ææ°ä¸¤æ¬¡è¯ä¼?    python compare_evals.py

    # æå®ä¸¤ä¸ªæä»¶å¯¹æ¯
    python compare_evals.py --file1 eval_results/sft_old.json --file2 eval_results/sft_new.json

    # æ¾ç¤ºå¼±é¡¹æ°æ®é?    python compare_evals.py --show_weak_datasets
"""
import os
import sys
import json
import argparse
import glob
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR.parent.parent)


def load_results(path: str) -> dict:
    """å è½½è¯ä¼°ç»æ JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_latest_files() -> tuple:
    """èªå¨æ¾å°ææ°çä¸¤ä¸ªè¯ä¼°æä»¶"""
    files = sorted(glob.glob("eval_results/*.json"))
    # æé¤ latest.json
    files = [f for f in files if not f.endswith("latest.json")]
    if len(files) < 2:
        print("â?éè¦è³å°?2 ä¸ªè¯ä¼°ç»ææä»¶æè½å¯¹æ¯?)
        print(f"   å½åæ¾å° {len(files)} ä¸ªæä»?)
        sys.exit(1)
    return files[-2], files[-1]


def compute_confidence_interval(scores, n_bootstrap: int = 1000, confidence: float = 0.95):
    """Bootstrap ç½®ä¿¡åºé´è®¡ç®"""
    import numpy as np
    if not scores or len(scores) < 5:
        return 0.0, 0.0, 1.0
    scores = np.array(scores, dtype=np.float32)
    n = len(scores)
    rng = np.random.RandomState(42)
    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        boot_means[i] = rng.choice(scores, size=n, replace=True).mean()
    alpha = 1 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    p_value = float((boot_means <= 0.0).mean())
    if p_value == 0.0:
        p_value = 1.0 / n_bootstrap
    return lower, upper, p_value


def compute_two_sample_pvalue(scores1, scores2, n_bootstrap: int = 1000):
    """éå¯¹ Bootstrap æ£éª?p-value"""
    import numpy as np
    if not scores1 or not scores2 or len(scores1) != len(scores2):
        return 1.0
    scores1 = np.array(scores1, dtype=np.float32)
    scores2 = np.array(scores2, dtype=np.float32)
    diff = scores2 - scores1
    rng = np.random.RandomState(42)
    n = len(diff)
    count = sum(1 for _ in range(n_bootstrap)
                if diff[rng.choice(n, size=n, replace=True)].mean() <= 0)
    return float(count / n_bootstrap)


def compare_results(r1: dict, r2: dict, show_weak: bool = False):
    """éææ å¯¹æ¯ä¸¤æ¬¡ç»æï¼å«ç½®ä¿¡åºé´åæ¾èæ§ï¼"""
    print("\n" + "=" * 90)
    print("ð è¯ä¼°ç»æå¯¹æ¯åæï¼å«ç»è®¡æ¾èæ§ï¼")
    print("=" * 90)

    # åä¿¡æ?    print(f"\nð çæ¬ä¿¡æ¯:")
    print(f"  æä»¶1: {r1.get('timestamp', 'N/A')} | stage={r1.get('stage', 'N/A')} | model={r1.get('model_path', 'N/A')}")
    print(f"  æä»¶2: {r2.get('timestamp', 'N/A')} | stage={r2.get('stage', 'N/A')} | model={r2.get('model_path', 'N/A')}")

    # æ°ç»æï¼datasets åµå¥
    if "datasets" in r1 and "datasets" in r2:
        _compare_datasets(r1["datasets"], r2["datasets"], show_weak)
    else:
        # æ§ç»æï¼æå¹³ keys
        _compare_flat(r1, r2)

    print()


def _compare_datasets(d1: dict, d2: dict, show_weak: bool):
    """å¯¹æ¯æ°ç datasets åµå¥ç»æ"""
    common_datasets = set(d1.keys()) & set(d2.keys())
    if not common_datasets:
        print("â ï¸ ä¸¤ä¸ªæä»¶æ²¡æå±åçæ°æ®é")
        return

    print(f"\nð æ°æ®éçº§ææ ååï¼{len(common_datasets)} ä¸ªå±åæ°æ®éï¼?")
    print(f"  {'æ°æ®é?:<20s} {'ææ ':<15s} {'æä»¶1':>8s} {'æä»¶2':>8s} {'åå':>8s} {'95%CI':>15s} {'p-value':>9s} {'è¶å¿'}")
    print(f"  {'-'*20} {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*15} {'-'*9} {'-'*10}")

    improved = 0
    regressed = 0
    weak_datasets = []

    for ds_name in sorted(common_datasets):
        ds1, ds2 = d1[ds_name], d2[ds_name]
        # æ¾åºæææ°å¼åææ 
        metrics = set()
        for r in [ds1, ds2]:
            metrics.update([k for k, v in r.items() if isinstance(v, (int, float))
                            and k not in ["total", "p_value", "lower", "upper"]])

        for metric in sorted(metrics):
            v1 = ds1.get(metric, 0)
            v2 = ds2.get(metric, 0)
            if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
                continue

            delta = v2 - v1
            # ä¼°ç®ç½®ä¿¡åºé´ï¼åºäºæ ·æ¬éï¼?            n1 = ds1.get("total", 100)
            n2 = ds2.get("total", 100)
            ci_half = 1.96 * (v1 * (1 - v1) / max(n1, 10)) ** 0.5
            ci_lo = v2 - ci_half
            ci_hi = v2 + ci_half

            # ç®åç p-value ä¼°è®¡
            se = ((v1 * (1 - v1) / max(n1, 10)) + (v2 * (1 - v2) / max(n2, 10))) ** 0.5
            z = abs(delta) / max(se, 1e-6)
            p_value = 2 * (1 - _norm_cdf(z))

            # è¶å¿å¤å®
            if abs(delta) < 0.01:
                trend = "â?æå¹³"
            elif p_value < 0.05 and delta > 0:
                trend = "â?æ¾èä¸å"
                improved += 1
            elif p_value < 0.05 and delta < 0:
                trend = "ð¨ æ¾èä¸é"
                regressed += 1
                weak_datasets.append((ds_name, metric, delta))
            elif delta > 0:
                trend = "ð¡ ç¥å(ä¸æ¾è?"
                improved += 1
            else:
                trend = "ð¡ ç¥é(ä¸æ¾è?"

            ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]"
            p_str = f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001"
            print(f"  {ds_name:<20s} {metric:<15s} {v1:>8.4f} {v2:>8.4f} {delta:>+8.4f} {ci_str:>15s} {p_str:>9s} {trend}")

    print(f"\nð æ»ç»: {improved} é¡¹ä¸åï¼å?{improved - regressed if improved > regressed else 0} æ¾èï¼ï¼{regressed} é¡¹ä¸é?)

    if regressed > 0:
        print(f"\nâ ï¸ éåçææ :")
        for ds, metric, delta in weak_datasets:
            print(f"   {ds}.{metric}: {delta:+.4f}")

    if show_weak and weak_datasets:
        print(f"\nð´ å¼±é¡¹æ°æ®éåè¡¨ï¼ç¨äº resample_data.py éç¹è®­ç»ï¼?")
        for ds, metric, delta in weak_datasets:
            print(f"   {ds}")


def _compare_flat(r1: dict, r2: dict):
    """å¯¹æ¯æ§çæå¹³ç»æ"""
    keys = set(list(r1.keys()) + list(r2.keys()))
    numeric_keys = [k for k in keys if isinstance(r1.get(k), (int, float)) or isinstance(r2.get(k), (int, float))]
    meta_keys = [k for k in keys if k not in numeric_keys and k not in ["timestamp", "stage", "model_path"]]

    print(f"\nð ææ åå:")
    print(f"  {'ææ ':<25s} {'æä»¶1':>10s} {'æä»¶2':>10s} {'åå':>10s} {'è¶å¿'}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    improved = 0
    regressed = 0
    for key in sorted(numeric_keys):
        v1 = r1.get(key, 0)
        v2 = r2.get(key, 0)
        if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
            continue
        delta = v2 - v1
        if abs(delta) < 0.001:
            trend = "â¡ï¸ æå¹³"
        elif delta > 0:
            trend = "â?ä¸å"
            improved += 1
        else:
            trend = "â ï¸ ä¸é"
            regressed += 1
        print(f"  {key:<25s} {v1:>10.4f} {v2:>10.4f} {delta:>+10.4f} {trend}")

    print(f"\nð æ»ç»: {improved} é¡¹ä¸å? {regressed} é¡¹ä¸é?)
    if regressed > improved:
        print(f"  â ï¸ éåææ å¤äºæ¹åææ ï¼å»ºè®®æ£æ¥è®­ç»éç½?)
    elif improved > regressed:
        print(f"  â?æ¨¡åè½åæ´ä½æå")
    else:
        print(f"  â¡ï¸ æ¨¡åè½åæ æ¾èåå?)


def _norm_cdf(x: float) -> float:
    """æ åæ­£æåå¸ç´¯ç§¯åå¸å½æ?""
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main():
    parser = argparse.ArgumentParser(description="å¯¹æ¯ä¸¤æ¬¡è¯ä¼°ç»æï¼åçº§çï¼?)
    parser.add_argument("--file1", type=str, default=None, help="ç¬¬ä¸ä¸ªè¯ä¼°æä»?)
    parser.add_argument("--file2", type=str, default=None, help="ç¬¬äºä¸ªè¯ä¼°æä»?)
    parser.add_argument("--show_weak_datasets", action="store_true",
                        help="æ¾ç¤ºå¼±é¡¹æ°æ®éåè¡?)
    args = parser.parse_args()

    if args.file1 and args.file2:
        f1, f2 = args.file1, args.file2
    else:
        f1, f2 = find_latest_files()

    print(f"ð å¯¹æ¯æä»¶:")
    print(f"  æä»¶1: {f1}")
    print(f"  æä»¶2: {f2}")

    r1 = load_results(f1)
    r2 = load_results(f2)
    compare_results(r1, r2, show_weak=args.show_weak_datasets)


if __name__ == "__main__":
    main()
