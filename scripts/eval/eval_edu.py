"""
QwenVL-Tutor 教育场景评估脚本

两阶段评估矩�?(SFT �?GRPO):
    baseline  : 基座模型 �?保存基线�?eval_results/baseline.json
    sft       : SFT �?�?全量评估 19 个数据集（确保新能力未退化）
    grpo      : GRPO�?�?ScienceQA保持检�?50) + 奖励质量(100) + 五维度细粒度 + 退化检�?
    full      : 最终发�?�?19 个评估集全量(500/集合) + ScienceQA test split 全量(4241)
    fine      : 细粒�?�?EduRewardModel 五维度评�?100) + ScienceQA参�?50)

测试集说�?
    ScienceQA  : validation split (日常迭代) / test split (最终holdout,全量4241�?
    C-Eval     : 5个保留学�?high_school_math/physics, college_physics/programming, discrete_math)
    19 个评估集: 7 个新�?we_math/geo170k/windata_math/cmmu/cmmmu/m3exam/mmscibench) + 12 个原�?
    自定义数�? : 本地 Parquet 文件 (�?dataset/edu_grpo.parquet)

> 2026-06 调整：移�?DPO 阶段评估（已去除 DPO 训练），新增 7 个新数据集评估注�?
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
from model.qwen_vlm import QwenVLTutor, QwenVLTutorConfig

warnings.filterwarnings('ignore')


def load_model(model_path, base_model_name="./model/Qwen2-VL-2B-Instruct",
              use_vllm=False, tensor_parallel_size=1, gpu_memory_utilization=0.85):
    """
    加载评估模型（支持 vLLM 加速 / HF transformers）

    Args:
        model_path: LoRA 路径或完整模型路径
        base_model_name: 基座模型路径
        use_vllm: 是否使用 vLLM 推理（True=快 5-20x，False=兼容）
        tensor_parallel_size: 张量并行 GPU 数
        gpu_memory_utilization: vLLM 显存使用率
    """
    if use_vllm:
        try:
            from scripts.eval.vllm_inference import VLLMBackend
            print(f"[vLLM] 加载模型 {model_path}（TP={tensor_parallel_size}）")
            backend = VLLMBackend(
                model_path=model_path,
                base_model_path=base_model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            return backend, "vllm"
        except Exception as e:
            print(f"[警告] vLLM 加载失败: {e}")
            print(f"[降级] 使用 HF transformers")
            use_vllm = False

    # HF transformers 后端
    config = QwenVLTutorConfig(
        model_name_or_path=base_model_name,
        use_lora=True,
    )
    model = QwenVLTutor(config)

    if os.path.exists(model_path):
        from peft import PeftModel
        model.base_model = PeftModel.from_pretrained(
            model.base_model, model_path
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, device


# 全局缓存模型（避免重复加载）
_MODEL_CACHE = {}


def get_or_load_model(model_path, base_model_name="./model/Qwen2-VL-2B-Instruct",
                      use_vllm=False, tensor_parallel_size=1):
    """获取或加载模型（全局缓存）"""
    key = (model_path, base_model_name, use_vllm, tensor_parallel_size)
    if key not in _MODEL_CACHE:
        model, backend = load_model(
            model_path, base_model_name,
            use_vllm=use_vllm,
            tensor_parallel_size=tensor_parallel_size,
        )
        _MODEL_CACHE[key] = (model, backend)
    return _MODEL_CACHE[key]


def evaluate_scienceqa(model, processor, device, max_samples=200, split="validation"):
    """
    �?ScienceQA 上评�?
    Args:
        split: "validation"(日常迭代) �?"test"(最终holdout)
    """
    from datasets import load_dataset

    print(f"加载 ScienceQA {split} �?..")
    ds = load_dataset("derek-thomas/ScienceQA", split=split)

    correct = 0
    total = 0
    has_steps = 0
    has_scaffolding = 0

    for item in tqdm(ds.select(range(min(len(ds), max_samples))), desc="评估�?):
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

        if any(kw in response for kw in ["观察", "思�?, "想一�?, "你能发现"]):
            has_scaffolding += 1

    # 输出结果
    print("\n" + "=" * 60)
    print(f"📊 ScienceQA 评估结果 ({split} split)")
    print("=" * 60)
    print(f"  总样本数:         {total}")
    print(f"  答案准确�?       {correct / max(total, 1) * 100:.1f}%")
    print(f"  步骤完整�?       {has_steps / max(total, 1) * 100:.1f}%")
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
    """�?C-Eval 中文理科验证集上评估（只取保留的未训练学科）"""
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
                if gt_answer in response[:50]:  # 只在开�?0字符里搜答案
                    correct += 1
                if any(kw in response for kw in ["步骤", "首先", "然后", "Step"]):
                    has_steps += 1
                if any(kw in response for kw in ["观察", "思�?, "想一�?]):
                    has_scaffolding += 1
            except Exception:
                continue
        if total >= max_samples:
            break

    print("\n" + "=" * 60)
    print("📊 C-Eval 中文理科评估结果")
    print("=" * 60)
    print(f"  总样本数:         {total}")
    print(f"  答案准确�?       {correct / max(total, 1) * 100:.1f}%")
    print(f"  步骤完整�?       {has_steps / max(total, 1) * 100:.1f}%")
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

    for idx in tqdm(indices, desc="评估�?):
        row = table.slice(idx, 1)
        conversations = json.loads(row['conversations'][0].as_py())
        image_bytes = row['image_bytes'][0].as_py()

        import io
        if isinstance(image_bytes, list):
            image_bytes = image_bytes[0]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 提取问题和答�?
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

        if any(kw in response for kw in ["观察", "思�?, "想一�?]):
            has_scaffolding += 1

    print("\n" + "=" * 60)
    print(f"📊 自定义数据集评估结果 ({data_path})")
    print("=" * 60)
    print(f"  总样本数:         {total}")
    print(f"  关键词匹配率:     {correct / max(total, 1) * 100:.1f}%")
    print(f"  步骤完整�?       {has_steps / max(total, 1) * 100:.1f}%")
    print(f"  启发式引导率:     {has_scaffolding / max(total, 1) * 100:.1f}%")
    print(f"  平均回答长度:     {sum(response_lengths) / max(len(response_lengths), 1):.0f} 字符")
    print("=" * 60)


def evaluate_grpo_reward(model, processor, device, data_path, max_samples=100):
    """
    GRPO 评估：用教育奖励模型评估生成的解答质�?
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

    for idx in tqdm(indices, desc="GRPO评估�?):
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
    print(f"  样本�?           {len(total_rewards)}")
    print(f"  平均奖励:         {avg_reward:.4f}  (target >0.5)")
    print(f"  最小奖�?         {min(total_rewards):.4f}")
    print(f"  最大奖�?         {max(total_rewards):.4f}")
    print(f"  奖励>0.5占比:     {sum(1 for r in total_rewards if r > 0.5) / max(len(total_rewards), 1) * 100:.1f}%")
    print("=" * 60)
    return avg_reward


def evaluate_regression(model, processor, device, baseline_path, max_samples=50):
    """检测基础能力是否退�?""
    import json as jslib
    if not os.path.exists(baseline_path):
        print(f"⚠️  基线文件不存�? {baseline_path}，跳过退化检�?)
        return True

    with open(baseline_path, 'r') as f:
        baseline = jslib.load(f)

    print("\n📋 退化检�? 对比训练�?ScienceQA 基线...")
    current = evaluate_scienceqa(model, processor, device, max_samples, split="validation")
    accuracy_drop = baseline.get('accuracy', 0) - current.get('accuracy', 0)
    step_drop = baseline.get('step_completeness', 0) - current.get('step_completeness', 0)

    print(f"\n  ScienceQA 准确率变�? {accuracy_drop*100:+.1f}%")
    print(f"  步骤完整率变�?       {step_drop*100:+.1f}%")

    if accuracy_drop > 0.10:
        print(f"  🚨 严重退化！准确率下�?>10%，建议回退模型或降低训练强�?)
        return False
    elif accuracy_drop > 0.05:
        print(f"  ⚠️  轻微退化，准确率下�?>5%，建议观察后续迭�?)
    else:
        print(f"  �?基础能力保持良好")
    return True


def evaluate_fine_grained(model, processor, device, data_path, max_samples=100):
    """
    细粒度评估：�?EduRewardModel 的五维度逐项评分
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

    for idx in tqdm(indices, desc="细粒度评�?):
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

        # 多维度评�?
        dims['accuracy'].append(reward_model._accuracy_score(response, gt_answer))
        dims['completeness'].append(reward_model._completeness_score(response))
        dims['fluency'].append(reward_model._fluency_score(response))
        dims['scaffolding'].append(reward_model._scaffolding_score(response))
        dims['format'].append(reward_model._format_score(response))

    print("\n" + "=" * 60)
    print("📊 细粒度维度评�?(五维度雷�?")
    print("=" * 60)
    labels = {'accuracy': '答案准确�?30%)', 'completeness': '步骤完整�?25%)',
              'fluency': '语言流畅�?15%)', 'scaffolding': '启发式引�?20%)', 'format': '格式规范(10%)'}
    for k, v in dims.items():
        avg = sum(v) / max(len(v), 1)
        bar = "�? * int(avg * 20) + "�? * (20 - int(avg * 20))
        print(f"  {labels[k]:18s} : {avg:.3f} [{bar}]")
    total = sum(sum(v)/max(len(v),1) * w for v, w in zip(dims.values(), [0.30, 0.25, 0.15, 0.20, 0.10]))
    print(f"  {'加权总分':18s} : {total:.3f}")
    print("=" * 60)


# 评估数据集注册表
# 根据训练数据集调整：加入 4 个中文核心图�?+ 3 个新中文多学科图文做题数据集
EVAL_DATASETS = {
    # === 中文核心图文数学（新增） ===
    'we_math': 'dataset/eval/eval_we_math.parquet',
    'geo170k': 'dataset/eval/eval_geo170k.parquet',
    'windata_math': 'dataset/eval/eval_windata_math.parquet',
    # === 中文多学科图文做题（新增�?===
    'cmmu': 'dataset/eval/eval_cmmu.parquet',
    'cmmmu': 'dataset/eval/eval_cmmmu.parquet',
    'm3exam': 'dataset/eval/eval_m3exam.parquet',
    'mmscibench': 'dataset/eval/eval_mmscibench.parquet',
    # === 原有 12 个评估集 ===
    'scienceqa': 'dataset/eval/eval_science.parquet',
    'ceval': None,  # 特殊处理：从 HF 动态加�?
    'ape210k': 'dataset/eval/eval_ape210k.parquet',
    'chartqa': 'dataset/eval/eval_chartqa.parquet',
    'cmmlu': 'dataset/eval/eval_cmmlu.parquet',
    'math_verse': 'dataset/eval/eval_math_verse.parquet',
    'math_vista': 'dataset/eval/eval_math_vista.parquet',
    'ocr': 'dataset/eval/eval_ocr.parquet',
    'openr1_math': 'dataset/eval/eval_openr1_math.parquet',
    'race': 'dataset/eval/eval_race.parquet',
    'gaokao_mathqa': 'dataset/eval/eval_gaokao_mathqa.parquet',
    'gaokao_mathcloze': 'dataset/eval/eval_gaokao_mathcloze.parquet',
}


def evaluate_dataset(model, processor, device, dataset_name, max_samples=200):
    """根据数据集名称选择对应的评估函�?""
    if dataset_name == 'ceval':
        return evaluate_ceval(model, processor, device, max_samples)
    
    data_path = EVAL_DATASETS.get(dataset_name)
    if data_path and os.path.exists(data_path):
        evaluate_custom(model, processor, device, data_path, max_samples)
    else:
        print(f"⚠️ 评估数据集不存在: {dataset_name} -> {data_path}")


def evaluate_all_datasets(model, processor, device, max_samples=200, save_raw_samples=False):
    """
    对所有已注册的评估数据集进行评估

    Args:
        save_raw_samples: 是否保存原始样本（含每条样本的判断结果）
                        保存�?results["datasets"][name]["raw_samples"]
                        用于�?
                        - 错误案例分析 (analyze_errors.py)
                        - 元评�?(meta_evaluation.py)
                        - 自定义指标研�?
    """
    from trainer.reward_model import EduRewardModel
    import pyarrow.parquet as pq
    import pyarrow as pa
    import io as iolib

    results = {}
    reward_model = EduRewardModel(tokenizer=processor.tokenizer) if save_raw_samples else None

    for name in sorted(EVAL_DATASETS.keys()):
        print(f"\n{'='*60}")
        print(f"📊 评估数据�? {name}")
        print(f"{'='*60}")
        try:
            if name == 'ceval':
                ds_result = evaluate_ceval(model, processor, device, max_samples,
                                            return_raw=save_raw_samples)
            else:
                path = EVAL_DATASETS[name]
                if path and os.path.exists(path):
                    ds_result = evaluate_custom(model, processor, device, path, max_samples,
                                                 return_raw=save_raw_samples)
                else:
                    print(f"  跳过（数据文件不存在�?)
                    continue

            # 收集 raw_samples
            if save_raw_samples and ds_result and reward_model:
                if "raw_samples" in ds_result:
                    # 限制每个数据集最多保�?100 条原始样本（节省空间�?
                    results[name] = {
                        "accuracy": ds_result.get("accuracy", 0),
                        "total": ds_result.get("total", 0),
                        "raw_samples": ds_result["raw_samples"][:100],
                    }
                else:
                    results[name] = ds_result
            else:
                results[name] = ds_result or {}
        except Exception as e:
            print(f"  �?评估失败: {e}")
            results[name] = {"error": str(e)}

    return results


from datetime import datetime


def save_eval_results(results: dict, stage: str, model_path: str):
    """
    持久化评估结果到带时间戳�?JSON 文件

    新结构（2026-06�?
        {
            "timestamp": "20260605_143000",
            "stage": "sft",
            "model_path": "out/edu_sft",
            "datasets": {  # 每个数据集的详细结果
                "scienceqa": {"accuracy": 0.723, "total": 932, "ci": {"accuracy": {"lower": 0.71, "upper": 0.74}}}
            },
            "aggregate": {  # 聚合指标
                "weighted_accuracy": 0.681,
                "weakest_datasets": [...]
            }
        }
    """
    os.makedirs("eval_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_results/{stage}_{timestamp}.json"

    # 自动附加置信区间（如果有原始分数�?
    if "datasets" in results:
        results = _attach_confidence_intervals(results)

    record = {
        "timestamp": timestamp,
        "stage": stage,
        "model_path": model_path,
        **results,
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    # 同时更新 latest 符号链接（用单独�?JSON 记录最新文件）
    latest_info = {"stage": stage, "timestamp": timestamp, "file": filename}
    with open("eval_results/latest.json", 'w', encoding='utf-8') as f:
        json.dump(latest_info, f, ensure_ascii=False)
    print(f"\n📁 评估结果已保�? {filename}")
    return filename


def _attach_confidence_intervals(results: dict) -> dict:
    """为评估结果附加置信区间（如果�?raw_scores 字段�?""
    if "datasets" not in results:
        return results

    for ds_name, ds_result in results["datasets"].items():
        raw = ds_result.get("raw_scores", [])
        if not raw:
            continue
        for metric in ["accuracy", "step_completeness", "scaffolding_rate"]:
            metric_raw = [r.get(metric, 0) for r in raw if metric in r]
            if metric_raw and len(metric_raw) >= 10:
                lo, hi, p = compute_confidence_interval(metric_raw)
                ds_result.setdefault("confidence_intervals", {})[metric] = {
                    "lower": round(lo, 4),
                    "upper": round(hi, 4),
                    "p_value": round(p, 4),
                }
        ds_result.pop("raw_scores", None)
    return results


def compute_confidence_interval(scores, n_bootstrap: int = 1000, confidence: float = 0.95):
    """
    使用 Bootstrap 方法计算分数列表的置信区间和 p-value

    Args:
        scores: 0/1 二元分数列表（每条样本是否正确）
        n_bootstrap: Bootstrap 采样次数
        confidence: 置信水平 (0.95 = 95%)

    Returns:
        (lower, upper, p_value) 元组
            - lower: 置信区间下界
            - upper: 置信区间上界
            - p_value: �?0 假设�?p-value
    """
    import numpy as np
    if not scores or len(scores) < 5:
        return 0.0, 0.0, 1.0

    scores = np.array(scores, dtype=np.float32)
    n = len(scores)
    mean = float(scores.mean())

    rng = np.random.RandomState(42)
    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 1 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    # p-value: 模型能力 > 0 的显著�?
    p_value = float((boot_means <= 0.0).mean())
    if p_value == 0.0:
        p_value = 1.0 / n_bootstrap

    return lower, upper, p_value


def compute_two_sample_pvalue(scores1, scores2, n_bootstrap: int = 1000):
    """
    计算两个分数列表差异�?p-value（配�?Bootstrap 检验）
    H0: 第二组不显著优于第一�?

    Args:
        scores1, scores2: 两次评估的二元分数列表（需等长�?
        n_bootstrap: Bootstrap 采样次数

    Returns:
        p_value: 第二组显著优于第一组的概率
    """
    import numpy as np
    if not scores1 or not scores2 or len(scores1) != len(scores2):
        return 1.0

    scores1 = np.array(scores1, dtype=np.float32)
    scores2 = np.array(scores2, dtype=np.float32)
    diff = scores2 - scores1

    rng = np.random.RandomState(42)
    n = len(diff)
    count = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if diff[idx].mean() <= 0:
            count += 1

    return float(count / n_bootstrap)


# ============================================================================
# CLI 入口
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QwenVL-Tutor 多阶段评�?)
    parser.add_argument("--model_path", type=str, default="./out/edu_sft", help="模型权重路径")
    parser.add_argument("--base_model", type=str, default="./model/Qwen2-VL-2B-Instruct", help="基座模型路径")
    parser.add_argument("--stage", type=str, default="sft",
                        choices=["baseline", "sft", "grpo", "full", "fine"],
                        help="评估阶段（已移除 dpo，采�?SFT→GRPO 两阶段）")
    parser.add_argument("--eval_data", type=str, default="dataset/edu_grpo.parquet", help="评估数据路径（GRPO奖励评估时使�?edu_grpo.parquet�?)
    parser.add_argument("--eval_dataset", type=str, default=None,
                        choices=list(EVAL_DATASETS.keys()),
                        help="评估单个数据集（新增�?)
    parser.add_argument("--eval_all", action="store_true", help="评估所有已注册数据集（新增�?)
    parser.add_argument("--max_samples", type=int, default=200, help="最大评估样本数�?1=全量�?)")
    parser.add_argument("--baseline_path", type=str, default="eval_results/baseline.json", help="退化检测基线文件路�?")
    # 🆕 vLLM 推理加速参数
    parser.add_argument("--use_vllm", action="store_true",
                        help="使用 vLLM 推理后端（5-20x 加速，需 CUDA 12.1+）")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="vLLM 张量并行 GPU 数（建议 1-4）")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                        help="vLLM 显存使用率（0.0-1.0）")
    args = parser.parse_args()

    setup_seed = lambda s: (random.seed(s), torch.manual_seed(s))
    setup_seed(42)

    model, device = load_model(
        args.model_path, args.base_model,
        use_vllm=args.use_vllm,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    processor = model.processor
    max_s = args.max_samples if args.max_samples > 0 else 99999

    # 新增：单独评估某个数据集
    if args.eval_dataset:
        print(f"🎯 单独评估: {args.eval_dataset}")
        evaluate_dataset(model, processor, device, args.eval_dataset, max_s)
        print(f"\n🏁 评估完成�?)
        exit(0)
    
    # 新增：评估所有数据集
    if args.eval_all:
        print(f"🎯 全量评估所�?{len(EVAL_DATASETS)} 个数据集")
        evaluate_all_datasets(model, processor, device, max_s)
        print(f"\n🏁 全量评估完成�?)
        exit(0)

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
        print(f"\n📁 基线已保�? {args.baseline_path}")
        save_eval_results(results, args.stage, args.model_path)

    elif args.stage == "sft":
        results = {}
        print(f"\n🔬 SFT 后全量评估（19 个数据集�?)
        # SFT 后全量评估所�?19 个评估集，确保新能力未退�?
        evaluate_all_datasets(model, processor, device, min(max_s, 200))
        # 重点：ScienceQA 核心指标记录到结果中
        r = evaluate_scienceqa(model, processor, device, min(max_s, 200), split="validation")
        results.update(r or {})
        results["stage_note"] = "SFT 后全量评�?19 个数据集"
        save_eval_results(results, args.stage, args.model_path)

    elif args.stage == "grpo":
        results = {}
        print(f"\n🔬 GRPO 后评估（SFT→GRPO 两阶段衔接）")
        print("\n📋 [1/4] ScienceQA 基础能力保持检�?..")
        r = evaluate_scienceqa(model, processor, device, min(max_s, 50), split="validation")
        results.update(r or {})
        print("\n📋 [2/4] GRPO 奖励质量评估�?K 精选）...")
        reward = evaluate_grpo_reward(model, processor, device, args.eval_data, min(max_s, 100))
        results["grpo_avg_reward"] = reward
        print("\n📋 [3/4] 细粒度五维度评估...")
        evaluate_fine_grained(model, processor, device, args.eval_data, min(max_s, 100))
        print("\n📋 [4/4] 退化检�?..")
        ok = evaluate_regression(model, processor, device, args.baseline_path, min(max_s, 50))
        results["regression_safe"] = ok
        save_eval_results(results, args.stage, args.model_path)

    elif args.stage == "full":
        # 全量评估�?9 个数据集 + ScienceQA test split + C-Eval holdout
        results = {}
        print(f"\n🔬 最终发布全量评�?({args.model_path})")
        print("\n📋 [1/2] 19 个评估集全量评估...")
        evaluate_all_datasets(model, processor, device, min(max_s, 500))
        print("\n📋 [2/2] ScienceQA test split (全量 holdout)...")
        r = evaluate_scienceqa(model, processor, device, max_s, split="test")
        results.update(r or {})
        results["stage_note"] = "最终发布评�?- 19 个评估集全量"
        save_eval_results(results, args.stage, args.model_path)

    elif args.stage == "fine":
        print("\n📋 细粒度多维度评估...")
        evaluate_fine_grained(model, processor, device, args.eval_data, min(max_s, 100))
        print("\n📋 基础 ScienceQA 参�?..")
        evaluate_scienceqa(model, processor, device, min(max_s, 50), split="validation")

    print(f"\n{'='*60}")
    print(f"🏁 评估完成�?)
    print(f"{'='*60}")
