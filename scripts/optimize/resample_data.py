"""
QwenSearch æ°æ®ééæ ·å·¥å?v2ï¼?026-06 éè®¾è®¡ï¼

æ°å¬å¼?
    base_weight = (1 - score) ** 0.5 + 0.5   # å¹³æ»å¼±é¡¹å¤ç»
    sample_correction = min(1.0, log10(n + 10) / 3.0)  # å°æ ·æ¬ææ?    weight = clamp(base_weight * sample_correction, 0.5, 2.5)

ç¸æ¯ v1 å¬å¼çä¼å?
    - å¼±é¡¹æéä¸ä¼è¿åº¦ï¼é¿åè¿æåï¼?    - å¼ºé¡¹ä¸ä¼è¢«è¿åº¦åå¼±ï¼ä¿çåºæ¬è½åï¼?    - å°æ°æ®éèªå¨ææ£ï¼é¿ååªå£°ä¸»å¯¼ï¼
    - å¹³æ»è¿æ¸¡ï¼æ æç«¯è·³å

ç¨æ³:
    python scripts/resample_data.py
    python scripts/resample_data.py --eval_file eval_results/sft_xxx.json
    python scripts/resample_data.py --output weights.json
    python scripts/resample_data.py --v1   # ä¸´æ¶ä½¿ç¨æ§å¬å¼ï¼å¯¹æ¯ç¨ï¼
"""
import os
import sys
import json
import argparse
import glob
import math
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR.parent.parent)


# è®­ç»æ°æ®éæ³¨åè¡¨ï¼?2 ä¸ªæ°æ®éï¼?TRAIN_DATASETS = {
    # ä¸­ææ ¸å¿å¾ææ°å­¦
    'we_math': 'dataset/edu_we_math.parquet',
    'geo170k': 'dataset/edu_geo170k.parquet',
    'windata_math': 'dataset/edu_windata_math.parquet',
    # ä¸­æå¤å­¦ç§å¾æåé¢?    'cmmu': 'dataset/edu_cmmu.parquet',
    'cmmmu': 'dataset/edu_cmmmu.parquet',
    'm3exam': 'dataset/edu_m3exam.parquet',
    'mmscibench': 'dataset/edu_mmscibench.parquet',
    # æ ¸å¿å¾ææ°å­¦
    'scienceqa': 'dataset/edu_science.parquet',
    'math_verse': 'dataset/edu_math_verse.parquet',
    'math_vista': 'dataset/edu_math_vista.parquet',
    # OCR + å¾è¡¨
    'ocr': 'dataset/edu_ocr.parquet',
    'chartqa': 'dataset/edu_chartqa.parquet',
    # ä¸­æçç§æ°å­¦
    'ceval': 'dataset/edu_ceval.parquet',
    'cmmlu': 'dataset/edu_cmmlu.parquet',
    'ape210k': 'dataset/edu_ape210k.parquet',
    'openr1_math': 'dataset/edu_openr1_math.parquet',
    'gaokao_mathqa': 'dataset/edu_gaokao_mathqa.parquet',
    'gaokao_mathcloze': 'dataset/edu_gaokao_mathcloze.parquet',
    # è¯­è¨çè§£
    'race': 'dataset/edu_race.parquet',
}


def load_eval_results(path: str) -> dict:
    """å è½½è¯ä¼°ç»æ"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_latest_eval() -> str:
    """èªå¨æ¾ææ°çè¯ä¼°ç»ææä»¶"""
    files = sorted(glob.glob("eval_results/*.json"))
    files = [f for f in files if not f.endswith("latest.json") and not f.endswith("audit_samples.json")]
    if not files:
        print("â?æªæ¾å°è¯ä¼°ç»ææä»¶ï¼è¯·åè¿è¡: python scripts/eval_edu.py --stage sft --eval_all")
        sys.exit(1)
    return files[-1]


def extract_dataset_scores(eval_data: dict) -> dict:
    """
    ä»è¯ä¼°ç»æä¸­æååæ°æ®éå¾å
    æ¯ææ°ç»æ?(datasets åµå¥) åæ§ç»æ (æå¹³ keys)
    """
    scores = {}

    # æ°ç»æ?    if "datasets" in eval_data:
        for ds_name, ds_result in eval_data["datasets"].items():
            acc = ds_result.get("accuracy")
            if acc is None:
                # å¼å®¹å¶ä»ææ å?                for k in ["step_completeness", "kw_match", "score"]:
                    if k in ds_result:
                        acc = ds_result[k]
                        break
            if acc is not None:
                scores[ds_name] = {
                    "score": float(acc),
                    "n_samples": ds_result.get("total", 100),
                }

    # æ§ç»æå¼å®¹ï¼accuracy_xxx / kw_match_xxx
    for key, value in eval_data.items():
        if key.startswith("accuracy_") or key.startswith("kw_match_"):
            ds_name = key.replace("accuracy_", "").replace("kw_match_", "")
            if ds_name not in scores:
                scores[ds_name] = {"score": float(value), "n_samples": 100}

    return scores


def compute_weight_v1(score: float, min_w: float = 0.3, max_w: float = 3.0) -> float:
    """v1 å¬å¼ï¼å·²å¼ç¨ï¼ä»ä½å¯¹æ¯ï¼"""
    adjust = 0.5 / max(score, 0.01)
    return max(min_w, min(max_w, adjust))


def compute_weight_v2(score: float, n_samples: int = 100,
                      min_w: float = 0.5, max_w: float = 2.5) -> float:
    """
    v2 å¬å¼ï¼æ¨èï¼

    Args:
        score: æ°æ®éå¾å?(0~1)
        n_samples: æ°æ®éæ ·æ¬æ°ï¼ç¨äºå°æ ·æ¬ä¿®æ­£ï¼?        min_w, max_w: æéä¸ä¸é?    """
    # 1. åºç¡æéï¼å¹³æ»å¼±é¡¹å¤ç»?    base_weight = (1.0 - score) ** 0.5 + 0.5  # score=0.5â?.21, score=0.1â?.45

    # 2. æ ·æ¬éä¿®æ­£ï¼æ ·æ¬é?< 100 çæ°æ®éï¼æéææ?    sample_correction = min(1.0, math.log10(n_samples + 10) / 3.0)

    weight = base_weight * sample_correction
    return max(min_w, min(max_w, weight))


def compute_resample_weights(eval_path: str, formula_version: str = "v2",
                             min_weight: float = 0.5, max_weight: float = 2.5) -> dict:
    """
    æ ¹æ®è¯ä¼°ç»æè®¡ç®ééæ ·æé?
    Args:
        eval_path: è¯ä¼°ç»ææä»¶è·¯å¾
        formula_version: "v1" (æ? æ?"v2" (æ°ï¼æ¨è)
        min_weight, max_weight: æéä¸ä¸é?    """
    eval_data = load_eval_results(eval_path)
    scores = extract_dataset_scores(eval_data)

    if not scores:
        print("â ï¸ è¯ä¼°ç»æä¸­æªæ¾å°æ°æ®éå¾åï¼ä½¿ç¨ååæé")
        return {k: 1.0 for k in TRAIN_DATASETS}

    weights = {}
    raw_weights = {}
    print("\n" + "=" * 80)
    print(f"ð åºäºè¯ä¼°åé¦çééæ ·æé (å¬å¼: {formula_version})")
    print("=" * 80)
    print(f"  {'æ°æ®é?:<20s} {'å¾å':>8s} {'æ ·æ¬é?:>8s} {'æé':>8s} {'å»ºè®®'}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")

    total_weight = 0
    for ds_name, ds_path in TRAIN_DATASETS.items():
        if not os.path.exists(ds_path):
            # æ°æ®éä¸å­å¨ï¼æéä¸º 0
            weights[ds_name] = 0.0
            continue

        # ä¼åä½¿ç¨è¯ä¼°æ°æ®ä¸­çå¾åï¼å¦åé»è®?0.5
        if ds_name in scores:
            score = scores[ds_name]["score"]
            n = scores[ds_name]["n_samples"]
        else:
            score = 0.5
            n = 100

        if formula_version == "v1":
            weight = compute_weight_v1(score, min_weight, max_weight)
        else:
            weight = compute_weight_v2(score, n, min_weight, max_weight)

        weights[ds_name] = round(weight, 3)
        raw_weights[ds_name] = weight
        total_weight += weight

        # å»ºè®®æ ç­¾
        hint = ""
        if weight > 1.8:
            hint = "ð´ éç¹è®­ç»"
        elif weight > 1.3:
            hint = "ð¡ å å¼ºè®­ç»"
        elif weight < 0.7:
            hint = "ð¢ éå½åå°"
        else:
            hint = "â?ä¿æ"

        print(f"  {ds_name:<20s} {score:>8.3f} {n:>8d} {weight:>8.3f} {hint}")

    print(f"\n  æéæ»å: {total_weight:.2f}  (åå: {len([w for w in weights.values() if w > 0]):.1f})")
    print("=" * 80)

    # åæ¶è¾åº v1 vs v2 å¯¹æ¯ï¼å¦ææ°æ®éé½è¯ä¼°äºï¼?    if formula_version == "v2" and scores:
        print(f"\nð v1 vs v2 å¬å¼å¯¹æ¯ï¼åä¸æ°æ®éï¼:")
        print(f"  {'æ°æ®é?:<20s} {'å¾å':>8s} {'v1æé':>8s} {'v2æé':>8s} {'åå':>10s}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
        for ds_name in sorted(scores.keys())[:10]:  # åªæ¾ç¤ºå 10 ä¸?            if ds_name in TRAIN_DATASETS:
                score = scores[ds_name]["score"]
                n = scores[ds_name]["n_samples"]
                w1 = compute_weight_v1(score)
                w2 = compute_weight_v2(score, n)
                delta = w2 - w1
                print(f"  {ds_name:<20s} {score:>8.3f} {w1:>8.3f} {w2:>8.3f} {delta:>+10.3f}")

    return weights


def generate_data_paths(weights: dict) -> str:
    """çæå¸¦æéç --data_paths åæ°å­ç¬¦ä¸?""
    paths = []
    for ds_name, weight in sorted(weights.items()):
        if weight > 0:
            paths.append(TRAIN_DATASETS[ds_name])
    return ",".join(paths)


def main():
    parser = argparse.ArgumentParser(description="è¯ä¼°åé¦æ°æ®ééæ ?v2")
    parser.add_argument("--eval_file", type=str, default=None, help="è¯ä¼°ç»æ JSON æä»¶è·¯å¾")
    parser.add_argument("--output", type=str, default=None, help="è¾åºæé JSON æä»¶")
    parser.add_argument("--v1", action="store_true", help="ä½¿ç¨ v1 æ§å¬å¼ï¼å¯¹æ¯ç¨ï¼")
    parser.add_argument("--min_weight", type=float, default=0.5, help="æéä¸é")
    parser.add_argument("--max_weight", type=float, default=2.5, help="æéä¸é")
    args = parser.parse_args()

    eval_path = args.eval_file or find_latest_eval()
    print(f"ð å è½½è¯ä¼°ç»æ: {eval_path}\n")

    formula = "v1" if args.v1 else "v2"
    weights = compute_resample_weights(
        eval_path,
        formula_version=formula,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )
    data_paths = generate_data_paths(weights)

    print(f"\nð æ¨èçè®­ç»å½ä»?")
    print(f"  python trainer/train_sft.py --data_paths \"{data_paths}\"")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                "eval_source": eval_path,
                "formula_version": formula,
                "weights": weights,
                "data_paths": data_paths,
            }, f, ensure_ascii=False, indent=2)
        print(f"\nð¾ æéå·²å¯¼å? {args.output}")


if __name__ == "__main__":
    main()
