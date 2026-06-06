"""
QwenVL-Tutor éè¯¯æ¡ä¾åæå·¥å·

åè½:
    1. èªå¨å½ç±»éè¯¯ç±»åï¼å¾åçè§?æ¨ç/æ ¼å¼/è®¡ç®/OCR/å¶ä»ï¼?    2. è¾åºéè¯¯åå¸æ¥å
    3. çææ°æ®è¡¥åå»ºè®®
    4. å¯¼åºéè¯¯æ ·æ¬ä¾äººå·¥æ ¸æ?
ç¨æ³:
    # åæææ°çè¯ä¼°ç»æ
    python analyze_errors.py

    # æå®è¯ä¼°æä»¶
    python analyze_errors.py --eval_file eval_results/sft_20260605.json

    # åæ¶å¯¼åºéè¯¯æ ·æ¬
    python analyze_errors.py --output_errors errors.json
"""
import os
import sys
import json
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR.parent.parent)


# éè¯¯ç±»åå³é®è¯?ERROR_PATTERNS = {
    "å¾åçè§£éè¯¯": [
        r"å¾ç.*?(çä¸æ¸|ä¸æ¸æ°|æ æ³è¯å«)",
        r"(æ²¡æ|ç¼ºå°).*?å¾å",
        r"å¾åçè§£å¤±è´¥",
        r"æªè¯å«å°",
    ],
    "æ¨çéè¯¯": [
        r"(æ¨ç|é»è¾|æè·¯).*?(éè¯¯|ä¸å¯¹|é?",
        r"å æ.*?éè¯¯",
        r"(åè®¾|åæ).*?(éè¯¯|ä¸æç«?",
    ],
    "ç­æ¡æ ¼å¼éè¯¯": [
        r"ç­æ¡.*?æ ¼å¼",
        r"æªæè¦æ±.*?ä½ç­",
        r"(ç¼ºå°|æ²¡æ).*?ç­æ¡",
        r"æªç»å?*?éé¡¹",
    ],
    "è®¡ç®éè¯¯": [
        r"(å |å|ä¹|é?.*?é?,
        r"è®¡ç®.*?éè¯¯",
        r"(ç®å¼|ç®é|ç®æ)",
    ],
    "OCRè¯å«éè¯¯": [
        r"(OCR|è¯å«).*?(å¤±è´¥|éè¯¯|ä¸åç¡?",
        r"æå­.*?è¯å«",
        r"(éå«å­|è¯å«æ|è¯¯è¯å?",
    ],
    "ç¥è¯éè¯¯": [
        r"æ¦å¿µ.*?(ä¸æ¸æ¥|éè¯¯|æ··æ·)",
        r"åç.*?(éè¯¯|ä¸å¯¹)",
        r"ä¸ç¥é?*?(å®ä¹|å¬å¼|å®ç)",
    ],
}


# éè¯¯ç±»å â?å»ºè®®æ°æ®è¡¥å
ERROR_TO_RECOMMENDATIONS = {
    "å¾åçè§£éè¯¯": "å¢å  OCR/è§è§çè§£ä¸é¡¹æ°æ®ï¼å¦ OCR-VQAãChartQAãDocVQAï¼?,
    "æ¨çéè¯¯": "å¢å æç»´é¾ï¼CoTï¼æ°æ®ï¼å¼ºè°åæ­¥æ¨ç",
    "ç­æ¡æ ¼å¼éè¯¯": "å¢å æ ¼å¼è§èè®­ç»æ°æ®ï¼æç¡?ç­æ¡æ?X"æ¨¡å¼ï¼?,
    "è®¡ç®éè¯¯": "å¢å è®¡ç®æ­¥éª¤æ°æ®ï¼å¼ºè°ä¸­é´è¿ç¨?,
    "OCRè¯å«éè¯¯": "å¢å ä¸­è±æ?OCR è®­ç»æ°æ®ï¼ç¹å«æ¯æåä½?,
    "ç¥è¯éè¯¯": "å¢å å­¦ç§åºç¡ç¥è¯æ°æ®ï¼å¦ææåé¢ãæ¦å¿µè§£éï¼",
}


def classify_error_type(response: str, gt_answer: str = "") -> str:
    """
    æ ¹æ®æ¨¡ååå¤åå®¹å¤æ­éè¯¯ç±»å

    Args:
        response: æ¨¡åçæçåå¤?        gt_answer: æ åç­æ¡

    Returns:
        éè¯¯ç±»åå­ç¬¦ä¸?    """
    if not response:
        return "å¶ä»éè¯¯"

    # 1. æ£æ¥åå¤ä¸­æ¯å¦åå«éè¯¯ç±»åå³é®è¯?    for error_type, patterns in ERROR_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, response):
                return error_type

    # 2. å¯åå¼ï¼åºäºåå¤ç¹å¾çå¤æ?    if "çä¸æ¸? in response or "å¾å" in response and "æ æ³" in response:
        return "å¾åçè§£éè¯¯"
    if len(response) < 20:
        return "ç­æ¡æ ¼å¼éè¯¯"
    if not re.search(r"ç­æ¡æ¯|ç­æ¡:|å æ­¤|æä»¥|æ?, response):
        return "æ¨çéè¯¯"
    return "å¶ä»éè¯¯"


def extract_wrong_samples(eval_file: str) -> list:
    """
    ä»è¯ä¼°æä»¶ä¸­æåéè¯¯æ ·æ¬
    æ³? éè¦è¯ä¼°ç»æä¸­åå« raw_samples å­æ®µ
    """
    with open(eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    wrong_samples = []
    datasets = data.get("datasets", {})

    for ds_name, ds_result in datasets.items():
        samples = ds_result.get("raw_samples", [])
        for s in samples:
            if not s.get("is_correct", True):  # éè¯¯æ ·æ¬
                wrong_samples.append({
                    "dataset": ds_name,
                    "question": s.get("question", "")[:200],
                    "gt_answer": s.get("gt_answer", "")[:100],
                    "model_response": s.get("response", "")[:300],
                    "error_type": classify_error_type(s.get("response", ""), s.get("gt_answer", "")),
                })
    return wrong_samples


def analyze_from_predictions(eval_file: str) -> dict:
    """
    ç´æ¥ä»è¯ä¼°æä»¶åæéè¯¯åå¸?    å¦æè¯ä¼°æä»¶æ²¡æ raw_samplesï¼åä½¿ç¨ confidence_intervals å?aggregate ä¿¡æ¯
    """
    with open(eval_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # å°è¯ä»?raw_samples æå
    if "datasets" in data:
        for ds_name, ds_result in data["datasets"].items():
            if "raw_samples" in ds_result:
                wrong_samples = extract_wrong_samples(eval_file)
                return _compute_distribution(wrong_samples)

    # åéï¼åºäºåç¡®çåç±»
    print("â ï¸ è¯ä¼°æä»¶ä¸­æªæ¾å° raw_samples å­æ®µ")
    print("   è¯·å¨è¯ä¼°æ¶å¯ç?--save_raw_samples éé¡¹ä¿å­åå§æ ·æ¬")
    print("   æèä½¿ç?--eval_file æåå«åå§æ ·æ¬çä¸­é´æä»¶")
    print()
    return _analyze_from_aggregate(data)


def _compute_distribution(wrong_samples: list) -> dict:
    """è®¡ç®éè¯¯ç±»ååå¸"""
    if not wrong_samples:
        return {"distribution": {}, "recommendations": [], "total": 0}

    counter = Counter([s["error_type"] for s in wrong_samples])
    total = len(wrong_samples)
    distribution = {
        etype: {"count": cnt, "ratio": round(cnt / total, 4)}
        for etype, cnt in counter.most_common()
    }

    # æ¶éå»ºè®®
    recommendations = []
    for etype in counter:
        if etype in ERROR_TO_RECOMMENDATIONS:
            recommendations.append({
                "error_type": etype,
                "count": counter[etype],
                "ratio": round(counter[etype] / total, 4),
                "recommendation": ERROR_TO_RECOMMENDATIONS[etype],
            })

    return {
        "distribution": distribution,
        "recommendations": recommendations,
        "total": total,
        "samples": wrong_samples[:50],  # æå¤ä¿ç?50 æ?    }


def _analyze_from_aggregate(data: dict) -> dict:
    """ä»?aggregate å­æ®µåæå¼±é¡¹æ°æ®é?""
    if "aggregate" not in data:
        return {"distribution": {}, "recommendations": [], "total": 0}

    weakest = data["aggregate"].get("weakest_datasets", [])
    recommendations = []
    for ds in weakest:
        recommendations.append({
            "dataset": ds,
            "recommendation": f"æ°æ®é?{ds} è¡¨ç°è¾å¼±ï¼å»ºè®®ï¼(1) å¢å è®­ç»æ ·æ¬æ°ï¼(2) æ£æ¥æ°æ®è´¨éï¼(3) è°æ´éæ ·æé",
        })
    return {
        "distribution": {},
        "recommendations": recommendations,
        "total": 0,
        "note": "ä»åºäº?aggregate å­æ®µçç®ååæï¼å»ºè®®å¯ç¨ raw_samples è·å¾è¯¦ç»éè¯¯ç±»ååå¸ï¼?,
    }


def print_report(analysis: dict, eval_file: str):
    """æå°éè¯¯åææ¥å"""
    print()
    print("=" * 70)
    print(f"ð éè¯¯æ¡ä¾åææ¥å")
    print(f"   è¯ä¼°æä»¶: {eval_file}")
    print("=" * 70)

    if analysis.get("note"):
        print(f"\nâ ï¸ {analysis['note']}\n")

    # éè¯¯åå¸
    distribution = analysis.get("distribution", {})
    total = analysis.get("total", 0)
    if distribution and total > 0:
        print(f"\nð éè¯¯ç±»ååå¸ï¼å± {total} æ¡éè¯¯æ ·æ¬ï¼:")
        max_count = max([d["count"] for d in distribution.values()]) if distribution else 1
        for etype, info in distribution.items():
            count = info["count"]
            ratio = info["ratio"]
            bar_len = int(count / max_count * 30)
            bar = "â? * bar_len + "â? * (30 - bar_len)
            print(f"  {etype:<15s} : {count:>4d} æ?({ratio*100:>5.1f}%)  {bar}")
    else:
        print("\nâ ï¸ æªåç°éè¯¯æ ·æ¬æ°æ?)

    # æ°æ®è¡¥åå»ºè®®
    recommendations = analysis.get("recommendations", [])
    if recommendations:
        print(f"\nð¡ æ°æ®è¡¥åå»ºè®®ï¼æä¼åçº§æåºï¼:")
        for i, rec in enumerate(recommendations, 1):
            if "error_type" in rec:
                print(f"  {i}. ã{rec['error_type']}ã?{rec['count']}æ? {rec['ratio']*100:.1f}%)")
                print(f"     â?{rec['recommendation']}")
            else:
                print(f"  {i}. ã{rec.get('dataset', '?')}ã?)
                print(f"     â?{rec['recommendation']}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="QwenVL-Tutor éè¯¯æ¡ä¾åæå·¥å·")
    parser.add_argument("--eval_file", type=str, default=None,
                        help="è¯ä¼°ç»æ JSON æä»¶ï¼é»è®¤ä½¿ç?eval_results/latest.jsonï¼?)
    parser.add_argument("--output_errors", type=str, default=None,
                        help="å¯¼åºéè¯¯æ ·æ¬å?JSON æä»¶")
    args = parser.parse_args()

    # ç¡®å®è¯ä¼°æä»¶
    if args.eval_file:
        eval_file = args.eval_file
    else:
        latest_info_path = "eval_results/latest.json"
        if not os.path.exists(latest_info_path):
            print("â?æªæ¾å°è¯ä¼°ç»æï¼è¯·åè¿è¡: python eval_edu.py --stage sft --eval_all")
            sys.exit(1)
        with open(latest_info_path, 'r', encoding='utf-8') as f:
            latest = json.load(f)
        eval_file = latest.get("file")
        if not eval_file or not os.path.exists(eval_file):
            print(f"â?è¯ä¼°æä»¶ä¸å­å? {eval_file}")
            sys.exit(1)

    # åæéè¯¯
    analysis = analyze_from_predictions(eval_file)
    print_report(analysis, eval_file)

    # å¯¼åºéè¯¯æ ·æ¬
    if args.output_errors and "samples" in analysis:
        with open(args.output_errors, 'w', encoding='utf-8') as f:
            json.dump(analysis["samples"], f, ensure_ascii=False, indent=2)
        print(f"\nð¾ éè¯¯æ ·æ¬å·²å¯¼å? {args.output_errors} ({len(analysis['samples'])} æ?")
    elif args.output_errors:
        print(f"\nâ ï¸ æ å¯å¯¼åºçéè¯¯æ ·æ¬ï¼å»ºè®®å¯ç¨ raw_samples éé¡¹ï¼?)


if __name__ == "__main__":
    main()
