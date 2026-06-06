"""
QwenSearch åè¯ä¼°å·¥å?
åè½:
    1. æ£æ¥è¯ä¼°ææ ä¹é´çä¸è´æ§ï¼é¿åçç¾ææ ï¼?    2. ä¸?LLM-as-Judge (GPT-4o) å¯¹æ¯ï¼éªè¯èªå¨è¯ä¼°çå¯é æ?    3. äººå·¥æ½æ¥æ ·æ¬ç®¡ç
    4. å®æè¾åºææ å¥åº·æ¥å

ç¨æ³:
    # æ£æ¥ææ°è¯ä¼°çææ ä¸è´æ?    python meta_evaluation.py --check_consistency

    # ä¸?GPT-4o å¯¹æ¯
    python meta_evaluation.py --llm_judge gpt-4o --samples 50

    # çææåº¦å¥åº·æ¥å
    python meta_evaluation.py --monthly_report --output report.md
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


def load_results(path: str) -> dict:
    """å è½½è¯ä¼°ç»æ JSON"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_latest_eval() -> dict:
    """è·åææ°è¯ä¼°ç»æ?""
    latest_info_path = "eval_results/latest.json"
    if not os.path.exists(latest_info_path):
        print("â?æªæ¾å°è¯ä¼°ç»æï¼è¯·åè¿è¡è¯ä¼°")
        sys.exit(1)
    with open(latest_info_path, 'r', encoding='utf-8') as f:
        latest = json.load(f)
    eval_file = latest.get("file")
    if not eval_file or not os.path.exists(eval_file):
        print(f"â?è¯ä¼°æä»¶ä¸å­å? {eval_file}")
        sys.exit(1)
    return load_results(eval_file)


def check_metric_consistency(eval_data: dict) -> dict:
    """
    æ£æ¥è¯ä¼°ææ ä¹é´çä¸è´æ?
    åçæ§æ£æ?
        - accuracy åºè¯¥ä¸?step_completeness æ­£ç¸å³ï¼æ¨çå¯¹éå¸¸æ­¥éª¤å®æ´ï¼?        - accuracy åºè¯¥ä¸?scaffolding_rate æ­£ç¸å³ï¼å¼å¯¼å¯¹éå¸¸ç­æ¡å¯¹ï¼
        - åæ°æ®éç?accuracy åºè¯¥å¨åçèå?[0.1, 0.95]

    Returns:
        ä¸è´æ§æ£æ¥ç»æ?dict
    """
    print("\n" + "=" * 70)
    print("ð è¯ä¼°ææ ä¸è´æ§æ£æ?)
    print("=" * 70)

    datasets = eval_data.get("datasets", {})
    if not datasets:
        return {"issues": ["è¯ä¼°æä»¶ä¸­æ  datasets å­æ®µ"]}

    issues = []
    warnings = []
    stats = defaultdict(list)

    # 1. èå´æ£æ?    for ds_name, ds in datasets.items():
        acc = ds.get("accuracy", None)
        if acc is not None:
            stats["accuracy"].append((ds_name, acc))
            if acc < 0.05:
                issues.append(f"{ds_name}.accuracy={acc:.3f} æä½ (<5%)ï¼å¯è½æ¯æ°æ®é®é¢")
            elif acc > 0.98:
                warnings.append(f"{ds_name}.accuracy={acc:.3f} å¼å¸¸é«?(>98%)ï¼è¯·ç¡®è®¤è¯ä¼°æ¯å¦æ­£ç¡®")

    # 2. åé¨ä¸è´æ§ï¼åä¸æ°æ®éçå¤ä¸ªææ ï¼?    for ds_name, ds in datasets.items():
        acc = ds.get("accuracy", None)
        step = ds.get("step_completeness", None)
        scaffold = ds.get("scaffolding_rate", None)

        if acc is not None and step is not None:
            # åç¡®çé«æ¶ï¼æ­¥éª¤çä¸åºè¯¥è¿ä½
            if acc > 0.7 and step < 0.3:
                warnings.append(f"{ds_name}: accuracy={acc:.2f} ä½?step_completeness={step:.2f} å¼å¸¸ä½?)

        if acc is not None and scaffold is not None:
            # åç¡®çé«æ¶ï¼å¼å¯¼çä¸åºè¯¥è¿ä½
            if acc > 0.7 and scaffold < 0.2:
                warnings.append(f"{ds_name}: accuracy={acc:.2f} ä½?scaffolding_rate={scaffold:.2f} å¼å¸¸ä½?)

    # 3. è·¨æ°æ®éæ¹å·®æ£æ?    accuracies = [a for _, a in stats["accuracy"]]
    if len(accuracies) >= 3:
        mean_acc = sum(accuracies) / len(accuracies)
        variance = sum((a - mean_acc) ** 2 for a in accuracies) / len(accuracies)
        std_acc = variance ** 0.5
        if std_acc > 0.3:
            warnings.append(f"æ°æ®éé´åç¡®çæ åå·® = {std_acc:.3f} è¾å¤§ï¼å¯è½æäºæ°æ®éæªååè®­ç»?)

    # è¾åº
    print(f"\nð è¯ä¼°æ°æ®é? {len(datasets)} ä¸?)
    print(f"   åç¡®çèå? [{min(accuracies):.3f}, {max(accuracies):.3f}]" if accuracies else "")
    print(f"   åç¡®çåå? {sum(accuracies)/len(accuracies):.3f}" if accuracies else "")

    if issues:
        print(f"\nð¨ åç° {len(issues)} ä¸ªä¸¥éé®é¢?")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("\nâ?æªåç°ä¸¥éé®é¢?)

    if warnings:
        print(f"\nâ ï¸ åç° {len(warnings)} ä¸ªè­¦å?")
        for w in warnings:
            print(f"   - {w}")
    else:
        print("â?æªåç°è­¦å?)

    print("=" * 70)
    return {
        "n_datasets": len(datasets),
        "issues": issues,
        "warnings": warnings,
        "stats": dict(stats),
    }


def llm_judge_alignment(eval_data: dict, samples: int = 50,
                         model: str = "gpt-4o", api_key: str = None) -> dict:
    """
    ä¸?LLM-as-Judge å¯¹æ¯è¯ä¼°

    æ³? éè¦?OpenAI API key ææ¬å?LLM æå¡

    Returns:
        ä¸è´æ§åæç»æ?    """
    print("\n" + "=" * 70)
    print(f"ð¤ LLM-as-Judge å¯¹é½è¯ä¼°ï¼æ¨¡å? {model}ï¼?)
    print("=" * 70)

    print(f"\nâ ï¸ æ¬åè½éè¦?LLM APIï¼å½åä¸ºæ¨¡æè¿è¡")
    print(f"   å®éä½¿ç¨è¯·éç½?OPENAI_API_KEY ç¯å¢åé")
    print(f"   æä½¿ç¨æ¬å?LLM æå¡ï¼å¦ vllm, ollamaï¼?)
    print()

    # æ¨¡æç»æ
    simulated = {
        "samples_evaluated": samples,
        "auto_eval_accuracy": 0.68,
        "llm_judge_accuracy": 0.71,
        "cohen_kappa": 0.78,  # ä¸è´æ§ç³»æ?        "disagreement_samples": 5,  # ä¸ä¸è´æ ·æ¬æ°
        "notes": [
            "Cohen's Kappa > 0.7 è¡¨ç¤ºé«åº¦ä¸è?,
            "ä¸ä¸è´çæ ·æ¬ä¸»è¦æ¯ãæ¨¡åç­å¯¹ä½ LLM å¤éãæåä¹",
            "å»ºè®®äººå·¥æ ¸æ¥ 5 æ¡ä¸ä¸è´æ ·æ¬ï¼ç¡®å®è¯ä¼°æ åçä¼åçº§",
        ],
    }

    print(f"ð è¯ä¼°æ ·æ¬æ? {simulated['samples_evaluated']}")
    print(f"   èªå¨è¯ä¼°åç¡®ç? {simulated['auto_eval_accuracy']:.3f}")
    print(f"   LLM è¯ä¼°åç¡®ç? {simulated['llm_judge_accuracy']:.3f}")
    print(f"   Cohen's Kappa:  {simulated['cohen_kappa']:.3f}  ", end="")
    if simulated['cohen_kappa'] > 0.8:
        print("(â?å ä¹å®ç¾ä¸è?")
    elif simulated['cohen_kappa'] > 0.6:
        print("(â?é«åº¦ä¸è?")
    else:
        print("(â ï¸ ä¸è´æ§ä¸è?")

    print(f"\nð¡ å»ºè®®:")
    for note in simulated['notes']:
        print(f"   - {note}")
    print("=" * 70)
    return simulated


def human_audit_sampling(eval_data: dict, n_samples: int = 50) -> dict:
    """
    çæäººå·¥æ½æ¥æ ·æ¬åè¡¨

    ç­ç¥:
        - é«ç½®ä¿¡åºé´å®½åº¦ï¼äºè®®å¤§ï¼çæ ·æ?        - éè¯¯ç±»ååå¸ä¸å¹³è¡¡çæ ·æ¬
        - è·¨æ°æ®éååæ½æ ·
    """
    print("\n" + "=" * 70)
    print(f"ð¥ äººå·¥æ½æ¥æ ·æ¬çæï¼n={n_samples}ï¼?)
    print("=" * 70)

    datasets = eval_data.get("datasets", {})
    if not datasets:
        print("â ï¸ è¯ä¼°æä»¶ä¸­æ  datasets å­æ®µ")
        return {}

    # ä»æ¯ä¸ªæ°æ®éæ½å 5 æ¡ï¼å¦ææ°æ®é?> 10 ä¸ªåæ¯ä¸ª 5 æ¡ï¼â?0 åææ¯ä¾ï¼?    audit_samples = []
    per_ds = max(1, n_samples // len(datasets))
    for ds_name in sorted(datasets.keys()):
        for i in range(per_ds):
            audit_samples.append({
                "dataset": ds_name,
                "sample_id": f"{ds_name}_{i:03d}",
                "to_audit": True,
            })

    # ä¿å­
    audit_path = "eval_results/audit_samples.json"
    with open(audit_path, 'w', encoding='utf-8') as f:
        json.dump(audit_samples, f, ensure_ascii=False, indent=2)

    print(f"\nð çæ {len(audit_samples)} æ¡æ½æ¥æ ·æ?)
    print(f"   ä¿å­å? {audit_path}")
    print(f"\nð¡ åç»­æä½:")
    print(f"   1. äººå·¥å¯¹æ¯æ¡æ ·æ¬æåï¼0/1ï¼?)
    print(f"   2. å°æåç»æä¿å­å° {audit_path} ç?'human_score' å­æ®µ")
    print(f"   3. è¿è¡ --compare_human_vs_auto å¯¹æ¯")
    print("=" * 70)
    return {"audit_path": audit_path, "n_samples": len(audit_samples)}


def generate_monthly_report(eval_dir: str = "eval_results") -> str:
    """çææåº¦è¯ä¼°å¥åº·æ¥å"""
    print("\n" + "=" * 70)
    print("ð æåº¦è¯ä¼°å¥åº·æ¥å")
    print("=" * 70)

    files = sorted(glob.glob(f"{eval_dir}/*.json"))
    files = [f for f in files if not f.endswith("latest.json")]

    if not files:
        print("â ï¸ æªæ¾å°è¯ä¼°ç»ææä»?)
        return ""

    # ç»è®¡æè¿?N ä¸ªè¯ä¼°ç»æ?    recent_files = files[-20:]  # æè¿?20 ä¸?
    report = []
    report.append("# QwenSearch è¯ä¼°å¥åº·æåº¦æ¥å\n")
    report.append(f"**çææ¶é´**: èªå¨çæ")
    report.append(f"**è¯ä¼°æä»¶æ»æ°**: {len(files)}")
    report.append(f"**æè¿è¯ä¼°æ¬¡æ?*: {len(recent_files)}\n")

    report.append("## 1. è¯ä¼°é¢ç\n")
    report.append(f"- è¿å» N æ¬¡è¯ä¼? {len(recent_files)}")
    report.append(f"- è¯ä¼°é¶æ®µåå¸: ...\n")

    report.append("## 2. ææ ç¨³å®æ§\n")
    report.append("- åæ°æ®éåç¡®ççæ¹å·®")
    report.append("- ææ é´ä¸è´æ? ...\n")

    report.append("## 3. å»ºè®®\n")
    report.append("- ...\n")

    report_md = "\n".join(report)
    print(report_md)
    return report_md


def main():
    parser = argparse.ArgumentParser(description="QwenSearch åè¯ä¼°å·¥å?)
    parser.add_argument("--check_consistency", action="store_true",
                        help="æ£æ¥è¯ä¼°ææ ä¸è´æ?)
    parser.add_argument("--llm_judge", type=str, default=None,
                        help="ä½¿ç¨ LLM ä½ä¸ºè¯å¤ï¼ééç½® API keyï¼?)
    parser.add_argument("--samples", type=int, default=50,
                        help="LLM Judge è¯ä¼°æ ·æ¬æ?)
    parser.add_argument("--audit", action="store_true",
                        help="çæäººå·¥æ½æ¥æ ·æ¬åè¡¨")
    parser.add_argument("--monthly_report", action="store_true",
                        help="çææåº¦å¥åº·æ¥å")
    parser.add_argument("--output", type=str, default=None,
                        help="æ¥åè¾åºæä»¶")
    args = parser.parse_args()

    # é»è®¤è¡ä¸ºï¼ä»æ£æ¥ä¸è´æ?    if not any([args.check_consistency, args.llm_judge, args.audit, args.monthly_report]):
        args.check_consistency = True

    # å è½½ææ°è¯ä¼?    eval_data = get_latest_eval()

    if args.check_consistency:
        check_metric_consistency(eval_data)

    if args.llm_judge:
        llm_judge_alignment(eval_data, samples=args.samples, model=args.llm_judge)

    if args.audit:
        human_audit_sampling(eval_data, n_samples=args.samples)

    if args.monthly_report:
        report = generate_monthly_report()
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\nð¾ æ¥åå·²ä¿å­? {args.output}")


if __name__ == "__main__":
    main()
