"""
QwenSearch 教育场景评估脚本

三阶段评估矩阵:
    baseline  : 基座模型 → 保存基线到 eval_results/baseline.json
    sft       : SFT 后 → ScienceQA(500) + C-Eval(200) + 自定义数据(200)
    dpo       : DPO 后 → ScienceQA保持检查(50) + 偏好gap(100) + 退化检测
    grpo      : GRPO后 → ScienceQA保持检查(50) + 奖励质量(100) + 退化检测
    full      : 最终发布 → ScienceQA test split 全量(4241) + C-Eval holdout(500)
    fine      : 细粒度 → EduRewardModel 五维度评分(100) + ScienceQA参考(50)

测试集说明:
    ScienceQA  : validation split (日常迭代) / test split (最终holdout,全量4241条)
    C-Eval     : 5个保留学科(high_school_math/physics, college_physics/programming, discrete_math)
    自定义数据  : 本地 Parquet 文件 (如 dataset/edu_science.parquet)
"""
import os
import sys
import argparse
import time
import json
import warnings
import torch
import random
from tqdm import tqdm
from PIL import Image
from model.qwen_vlm import QwenSearchVLM, QwenSearchConfig

warnings.filterwarnings('ignore')


def load_model(model_path, base_model_name="./model/Qwen2-VL-2B-Instruct"):
    """加载评估模型"""
    config = QwenSearchConfig(
        model_name_or_path=base_model_name,
        use_lora=True,
    )
    model = QwenSearchVLM(config)

    if os.path.exists(model_path):
        from peft import PeftModel
        model.base_model = PeftModel.from_pretrained(
            model.base_model, model_path
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, device


def evaluate_scienceqa(model, processor, device, max_samples=200, split="validation"):
    """
    在 ScienceQA 上评估
    Args:
        split: "validation"(日常迭代) 或 "test"(最终holdout)
    """
    from datasets import load_dataset

    print(f"加载 ScienceQA {split} 集...")
    ds = load_dataset("derek-thomas/ScienceQA", split=split)

    correct = 0
    total = 0
    has_steps = 0
    has_scaffolding = 0

    for item in tqdm(ds.select(range(min(len(ds), max_samples))), desc="评估中"):
        image = item.get('image')
        if image is None:
            continue

        question = item.get('question', '')
        choices = item.get('choices', [])
        if choices:
            choice_text = "\n".join(
                f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
            )
            question = f"{question}\n{choice_text}"

        answer_idx = item.get('answer', 0)
        gt_answer = chr(65 + answer_idx)

        # 构建消息
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"请解答这道题目，给出正确选项和解析：\n{question}"},
            ],
        }]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[text], images=[image],
            return_tensors="pt", padding=True,
        ).to(device)

        with torch.no_grad():
            generated_ids = model.base_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        response_ids = generated_ids[0][inputs.input_ids.shape[1]:]
        response = processor.tokenizer.decode(response_ids, skip_special_tokens=True)

        # 评估指标
        total += 1
        if gt_answer in response:
            correct += 1

        if any(kw in response for kw in ["步骤", "首先", "然后", "Step"]):
            has_steps += 1

        if any(kw in response for kw in ["观察", "思考", "想一想", "你能发现"]):
            has_scaffolding += 1

    # 输出结果
    print("\n" + "=" * 60)
    print(f"📊 ScienceQA 评估结果 ({split} split)")
    print("=" * 60)
    print(f"  总样本数:         {total}")
    print(f"  答案准确率:       {correct / max(total, 1) * 100:.1f}%")
    print(f"  步骤完整率:       {has_steps / max(total, 1) * 100:.1f}%")
    print(f"  启发式引导率:     {has_scaffolding / max(total, 1) * 100:.1f}%")
    print("=" * 60)

    return {
        "dataset": f"ScienceQA-{split}",
        "total": total,
        "accuracy": correct / max(total, 1),
        "step_completeness": has_steps / max(total, 1),
        "scaffolding_rate": has_scaffolding / max(total, 1),
    }


def evaluate_ceval(model, processor, device, max_samples=200):
    """在 C-Eval 中文理科验证集上评估（只取保留的未训练学科）"""
    from datasets import load_dataset

    # 用保留的理科子集做评估（不重复训练数据）
    EVAL_CONFIGS = [
        'high_school_mathematics', 'high_school_physics', 'college_physics',
        'college_programming', 'discrete_mathematics',
    ]
    correct = 0
    total = 0
    has_steps = 0
    has_scaffolding = 0

    for cfg in EVAL_CONFIGS:
        try:
            ds = load_dataset("ceval/ceval-exam", cfg, split="test", streaming=True)
        except Exception:
            continue
        for item in ds:
            if total >= max_samples:
                break
            try:
                question = item.get('question', '')
                choices = [item.get(k, '') for k in ['A', 'B', 'C', 'D'] if item.get(k)]
                if choices:
                    question += "\n" + "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
                gt_answer = str(item.get('answer', ''))

                messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": f"请解答这道选择题，给出正确选项和解析：\n{question}"}],
                }]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                inputs = processor(
                    text=[text], return_tensors="pt", padding=True,
                ).to(device)

                with torch.no_grad():
                    generated_ids = model.base_model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        pixel_values=inputs.pixel_values,
                        image_grid_thw=inputs.image_grid_thw,
                        max_new_tokens=256, do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                    )
                response_ids = generated_ids[0][inputs.input_ids.shape[1]:]
                response = processor.tokenizer.decode(response_ids, skip_special_tokens=True)

                total += 1
                if gt_answer in response[:50]:  # 只在开头50字符里搜答案
                    correct += 1
                if any(kw in response for kw in ["步骤", "首先", "然后", "Step"]):
                    has_steps += 1
                if any(kw in response for kw in ["观察", "思考", "想一想"]):
                    has_scaffolding += 1
            except Exception:
                continue
        if total >= max_samples:
            break

    print("\n" + "=" * 60)
    print("📊 C-Eval 中文理科评估结果")
    print("=" * 60)
    print(f"  总样本数:         {total}")
    print(f"  答案准确率:       {correct / max(total, 1) * 100:.1f}%")
    print(f"  步骤完整率:       {has_steps / max(total, 1) * 100:.1f}%")
    print(f"  启发式引导率:     {has_scaffolding / max(total, 1) * 100:.1f}%")
    print("=" * 60)


def evaluate_custom(model, processor, device, data_path, max_samples=200):
    """在自定义数据集上评估"""
    import pyarrow.parquet as pq
    import pyarrow as pa

    print(f"加载自定义数据集: {data_path}")
    table = pa.Table.from_batches(pq.ParquetFile(data_path).iter_batches())

    correct = 0
    total = 0
    has_steps = 0
    has_scaffolding = 0
    response_lengths = []

    indices = random.sample(range(len(table)), min(len(table), max_samples))

    for idx in tqdm(indices, desc="评估中"):
        row = table.slice(idx, 1)
        conversations = json.loads(row['conversations'][0].as_py())
        image_bytes = row['image_bytes'][0].as_py()

        import io
        if isinstance(image_bytes, list):
            image_bytes = image_bytes[0]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 提取问题和答案
        question = ""
        gt_answer = ""
        for turn in conversations:
            if turn['role'] == 'user':
                question = turn['content'].replace('<image>', '').strip()
            elif turn['role'] == 'assistant':
                gt_answer = turn['content']

        if not question:
            continue

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"请解答这道题目：\n{question}"},
            ],
        }]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[text], images=[image],
            return_tensors="pt", padding=True,
        ).to(device)

        with torch.no_grad():
            generated_ids = model.base_model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        response_ids = generated_ids[0][inputs.input_ids.shape[1]:]
        response = processor.tokenizer.decode(response_ids, skip_special_tokens=True)

        total += 1
        response_lengths.append(len(response))

        # 简单评估：检查回复中是否包含 GT 中的关键信息
        gt_kw = set(gt_answer.replace('\n', ' ').split()[:10])
        resp_kw = set(response.replace('\n', ' ').split()[:20])
        if len(gt_kw & resp_kw) > 0:
            correct += 1

        if any(kw in response for kw in ["步骤", "首先", "然后", "Step"]):
            has_steps += 1

        if any(kw in response for kw in ["观察", "思考", "想一想"]):
            has_scaffolding += 1

    print("\n" + "=" * 60)
    print(f"📊 自定义数据集评估结果 ({data_path})")
    print("=" * 60)
    print(f"  总样本数:         {total}")
    print(f"  关键词匹配率:     {correct / max(total, 1) * 100:.1f}%")
    print(f"  步骤完整率:       {has_steps / max(total, 1) * 100:.1f}%")
    print(f"  启发式引导率:     {has_scaffolding / max(total, 1) * 100:.1f}%")
    print(f"  平均回答长度:     {sum(response_lengths) / max(len(response_lengths), 1):.0f} 字符")
    print("=" * 60)


def evaluate_dpo_quality(model, processor, device, data_path, max_samples=100):
    """
    DPO 评估：对比模型在chosen vs rejected风格上的得分差异
    - 高分：模型明显偏好完整回答（chosen风格）
    - 低分：模型仍可能输出简短回答（rejected风格）
    """
    from trainer.reward_model import EduRewardModel
    import pyarrow.parquet as pq
    import pyarrow as pa
    import io as iolib

    reward_model = EduRewardModel(tokenizer=processor.tokenizer)
    print(f"加载 DPO 评估数据: {data_path}")
    table = pa.Table.from_batches(pq.ParquetFile(data_path).iter_batches())

    indices = random.sample(range(len(table)), min(len(table), max_samples))
    long_rewards = []  # 完整回答（chosen风格）的reward
    short_rewards = []  # 简短回答（rejected风格）的reward

    for idx in tqdm(indices, desc="DPO评估中"):
        row = table.slice(idx, 1)
        conversations = json.loads(row['conversations'][0].as_py())
        image_bytes = row['image_bytes'][0].as_py()
        if isinstance(image_bytes, list):
            image_bytes = image_bytes[0]
        image = Image.open(iolib.BytesIO(image_bytes)).convert("RGB")

        question = ""
        gt_answer = ""
        for turn in conversations:
            if turn['role'] == 'user':
                question = turn['content'].replace('<image>', '').strip()
            elif turn['role'] == 'assistant':
                gt_answer = turn['content']
        if not question:
            continue

        # 1. 正常温度生成（模拟chosen风格）
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"请详细解答这道题目，包含完整的解题步骤和解析：\n{question}"},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen_ids = model.base_model.generate(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values, image_grid_thw=inputs.image_grid_thw,
                max_new_tokens=256, temperature=0.7, top_p=0.95, do_sample=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        response_long = processor.tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        long_rewards.append(reward_model.compute_reward(response_long, gt_answer))

        # 2. 高温生成（模拟rejected风格）
        with torch.no_grad():
            gen_ids2 = model.base_model.generate(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values, image_grid_thw=inputs.image_grid_thw,
                max_new_tokens=64, temperature=1.2, top_p=0.85, do_sample=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        response_short = processor.tokenizer.decode(gen_ids2[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        short_rewards.append(reward_model.compute_reward(response_short, gt_answer))

    avg_long = sum(long_rewards) / max(len(long_rewards), 1)
    avg_short = sum(short_rewards) / max(len(short_rewards), 1)
    gap = avg_long - avg_short

    print("\n" + "=" * 60)
    print("📊 DPO 偏好质量评估")
    print("=" * 60)
    print(f"  样本数:           {len(long_rewards)}")
    print(f"  完整回答平均reward: {avg_long:.4f}  (chosen风格，target >0.5)")
    print(f"  简短回答平均reward: {avg_short:.4f}  (rejected风格)")
    print(f"  偏好差距:         {gap:+.4f}  (越大越好，代表DPO有效)")
    print(f"  DPO有效性判定:    {'✅ 有效 (gap>0.05)' if gap > 0.05 else '⚠️ 差距不足'}")
    print("=" * 60)
    return gap


def evaluate_grpo_reward(model, processor, device, data_path, max_samples=100):
    """
    GRPO 评估：用教育奖励模型评估生成的解答质量
    输出各维度的平均得分
    """
    from trainer.reward_model import EduRewardModel
    import pyarrow.parquet as pq
    import pyarrow as pa
    import io as iolib

    reward_model = EduRewardModel(tokenizer=processor.tokenizer)
    print(f"加载 GRPO 评估数据: {data_path}")
    table = pa.Table.from_batches(pq.ParquetFile(data_path).iter_batches())

    indices = random.sample(range(len(table)), min(len(table), max_samples))
    total_rewards = []

    for idx in tqdm(indices, desc="GRPO评估中"):
        row = table.slice(idx, 1)
        conversations = json.loads(row['conversations'][0].as_py())
        image_bytes = row['image_bytes'][0].as_py()
        if isinstance(image_bytes, list):
            image_bytes = image_bytes[0]
        image = Image.open(iolib.BytesIO(image_bytes)).convert("RGB")

        question = ""
        gt_answer = ""
        for turn in conversations:
            if turn['role'] == 'user':
                question = turn['content'].replace('<image>', '').strip()
            elif turn['role'] == 'assistant':
                gt_answer = turn['content']
        if not question:
            continue

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"请解答这道题目，给出解析和答案：\n{question}"},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            gen_ids = model.base_model.generate(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values, image_grid_thw=inputs.image_grid_thw,
                max_new_tokens=256, temperature=0.7, do_sample=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        response = processor.tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        r = reward_model.compute_reward(response, gt_answer)
        total_rewards.append(r)

    avg_reward = sum(total_rewards) / max(len(total_rewards), 1)

    print("\n" + "=" * 60)
    print("📊 GRPO 奖励模型评估")
    print("=" * 60)
    print(f"  样本数:           {len(total_rewards)}")
    print(f"  平均奖励:         {avg_reward:.4f}  (target >0.5)")
    print(f"  最小奖励:         {min(total_rewards):.4f}")
    print(f"  最大奖励:         {max(total_rewards):.4f}")
    print(f"  奖励>0.5占比:     {sum(1 for r in total_rewards if r > 0.5) / max(len(total_rewards), 1) * 100:.1f}%")
    print("=" * 60)
    return avg_reward


def evaluate_regression(model, processor, device, baseline_path, max_samples=50):
    """检测基础能力是否退化"""
    import json as jslib
    if not os.path.exists(baseline_path):
        print(f"⚠️  基线文件不存在: {baseline_path}，跳过退化检测")
        return True

    with open(baseline_path, 'r') as f:
        baseline = jslib.load(f)

    print("\n📋 退化检测: 对比训练前 ScienceQA 基线...")
    current = evaluate_scienceqa(model, processor, device, max_samples, split="validation")
    accuracy_drop = baseline.get('accuracy', 0) - current.get('accuracy', 0)
    step_drop = baseline.get('step_completeness', 0) - current.get('step_completeness', 0)

    print(f"\n  ScienceQA 准确率变化: {accuracy_drop*100:+.1f}%")
    print(f"  步骤完整率变化:       {step_drop*100:+.1f}%")

    if accuracy_drop > 0.10:
        print(f"  🚨 严重退化！准确率下降 >10%，建议回退模型或降低训练强度")
        return False
    elif accuracy_drop > 0.05:
        print(f"  ⚠️  轻微退化，准确率下降 >5%，建议观察后续迭代")
    else:
        print(f"  ✅ 基础能力保持良好")
    return True


def evaluate_fine_grained(model, processor, device, data_path, max_samples=100):
    """
    细粒度评估：用 EduRewardModel 的五维度逐项评分
    输出雷达图风格的维度打分
    """
    from trainer.reward_model import EduRewardModel
    import pyarrow.parquet as pq
    import pyarrow as pa
    import io as iolib

    reward_model = EduRewardModel(tokenizer=processor.tokenizer)
    table = pa.Table.from_batches(pq.ParquetFile(data_path).iter_batches())
    indices = random.sample(range(len(table)), min(len(table), max_samples))

    dims = {'accuracy': [], 'completeness': [], 'fluency': [], 'scaffolding': [], 'format': []}

    for idx in tqdm(indices, desc="细粒度评估"):
        row = table.slice(idx, 1)
        conversations = json.loads(row['conversations'][0].as_py())
        image_bytes = row['image_bytes'][0].as_py()
        if isinstance(image_bytes, list): image_bytes = image_bytes[0]
        image = Image.open(iolib.BytesIO(image_bytes)).convert("RGB")

        question = ""; gt_answer = ""
        for t in conversations:
            if t['role'] == 'user': question = t['content'].replace('<image>', '').strip()
            elif t['role'] == 'assistant': gt_answer = t['content']
        if not question: continue

        # 生成回答
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"请解答这道题目：\n{question}"},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen_ids = model.base_model.generate(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values, image_grid_thw=inputs.image_grid_thw,
                max_new_tokens=256, temperature=0.7, do_sample=True,
                pad_token_id=processor.tokenizer.pad_token_id, eos_token_id=processor.tokenizer.eos_token_id,
            )
        response = processor.tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # 多维度评分
        dims['accuracy'].append(reward_model._accuracy_score(response, gt_answer))
        dims['completeness'].append(reward_model._completeness_score(response))
        dims['fluency'].append(reward_model._fluency_score(response))
        dims['scaffolding'].append(reward_model._scaffolding_score(response))
        dims['format'].append(reward_model._format_score(response))

    print("\n" + "=" * 60)
    print("📊 细粒度维度评分 (五维度雷达)")
    print("=" * 60)
    labels = {'accuracy': '答案准确性(30%)', 'completeness': '步骤完整性(25%)',
              'fluency': '语言流畅度(15%)', 'scaffolding': '启发式引导(20%)', 'format': '格式规范(10%)'}
    for k, v in dims.items():
        avg = sum(v) / max(len(v), 1)
        bar = "█" * int(avg * 20) + "░" * (20 - int(avg * 20))
        print(f"  {labels[k]:18s} : {avg:.3f} [{bar}]")
    total = sum(sum(v)/max(len(v),1) * w for v, w in zip(dims.values(), [0.30, 0.25, 0.15, 0.20, 0.10]))
    print(f"  {'加权总分':18s} : {total:.3f}")
    print("=" * 60)


# ============================================================================
# CLI 入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenSearch 多阶段评估")
    parser.add_argument("--model_path", type=str, default="./out/edu_sft", help="模型权重路径")
    parser.add_argument("--base_model", type=str, default="./model/Qwen2-VL-2B-Instruct", help="基座模型路径")
    parser.add_argument("--stage", type=str, default="sft",
                        choices=["baseline", "sft", "dpo", "grpo", "full", "fine"],
                        help="评估阶段")
    parser.add_argument("--eval_data", type=str, default="dataset/edu_science.parquet", help="评估数据路径")
    parser.add_argument("--max_samples", type=int, default=200, help="最大评估样本数（-1=全量）")
    parser.add_argument("--baseline_path", type=str, default="eval_results/baseline.json", help="退化检测基线文件路径")
    args = parser.parse_args()

    setup_seed = lambda s: (random.seed(s), torch.manual_seed(s))
    setup_seed(42)

    model, device = load_model(args.model_path, args.base_model)
    processor = model.processor
    max_s = args.max_samples if args.max_samples > 0 else 99999

    if args.stage == "baseline":
        # 基座评估 + 保存基线
        print("🔍 基座模型基线评估...")
        results = {"accuracy": 0, "step_completeness": 0, "scaffolding_rate": 0}
        print("\n📋 [1/2] ScienceQA 基线...")
        r = evaluate_scienceqa(model, processor, device, max_s, split="validation")
        results.update(r)
        print("\n📋 [2/2] C-Eval 中文基线...")
        evaluate_ceval(model, processor, device, min(max_s, 200))
        # 保存基线
        os.makedirs("eval_results", exist_ok=True)
        with open(args.baseline_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n📁 基线已保存: {args.baseline_path}")

    elif args.stage == "sft":
        print("\n📋 [1/3] ScienceQA 英文图文评估...")
        evaluate_scienceqa(model, processor, device, max_s, split="validation")
        print("\n📋 [2/3] C-Eval 中文理科评估...")
        evaluate_ceval(model, processor, device, min(max_s, 200))
        print("\n📋 [3/3] 自定义数据评估...")
        evaluate_custom(model, processor, device, args.eval_data, min(max_s, 200))

    elif args.stage == "dpo":
        print("\n📋 [1/3] ScienceQA 基础能力保持检查...")
        evaluate_scienceqa(model, processor, device, min(max_s, 50), split="validation")
        print("\n📋 [2/3] DPO 偏好质量评估...")
        evaluate_dpo_quality(model, processor, device, args.eval_data, min(max_s, 100))
        print("\n📋 [3/3] 退化检测...")
        evaluate_regression(model, processor, device, args.baseline_path, min(max_s, 50))

    elif args.stage == "grpo":
        print("\n📋 [1/3] ScienceQA 基础能力保持检查...")
        evaluate_scienceqa(model, processor, device, min(max_s, 50), split="validation")
        print("\n📋 [2/3] GRPO 奖励质量评估...")
        evaluate_grpo_reward(model, processor, device, args.eval_data, min(max_s, 100))
        print("\n📋 [3/3] 退化检测...")
        evaluate_regression(model, processor, device, args.baseline_path, min(max_s, 50))

    elif args.stage == "full":
        # 全量评估：ScienceQA test split + C-Eval holdout
        print(f"\n🔬 最终发布全量评估 ({args.model_path})")
        print("\n📋 [1/2] ScienceQA test split (全量 holdout)...")
        evaluate_scienceqa(model, processor, device, max_s, split="test")
        print("\n📋 [2/2] C-Eval holdout 学科...")
        evaluate_ceval(model, processor, device, min(max_s, 500))

    elif args.stage == "fine":
        print("\n📋 细粒度多维度评估...")
        evaluate_fine_grained(model, processor, device, args.eval_data, min(max_s, 100))
        print("\n📋 基础 ScienceQA 参考...")
        evaluate_scienceqa(model, processor, device, min(max_s, 50), split="validation")

    print(f"\n{'='*60}")
    print(f"🏁 评估完成！")
    print(f"{'='*60}")