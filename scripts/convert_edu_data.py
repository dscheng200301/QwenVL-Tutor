"""
教育数据集格式转换工具
将多个开源教育数据集统一转换为 QwenVL-Tutor 的 Parquet 格式

支持的数据集（共23个）:
    图文数学: scienceqa, mathverse, mathvista, tabmwp, math23k, ape210k, geometry3k, clevr_math
    图文图表: chartqa, dvqa
    中文综合: ceval, cmmlu, gaokao, geoqa
    科学常识: ai2d, biology, tqa
    图文通识: mmmu, ocr_vqa, vizwiz
    语言理解: race

输出格式:
    Parquet 文件，包含列:
    - conversations: JSON string
    - image_bytes: binary (JPEG 编码)
    - subject: str (可选)
    - grade_level: str (可选)
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import io
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# 教育场景 System Prompt
EDU_SYSTEM_PROMPT = (
    "你是一位耐心的辅导老师，请仔细查看题目图片，"
    "理解题意后给出正确的解答。请包含以下内容：\n"
    "1. 题目分析：简要说明题目考查的知识点\n"
    "2. 解题步骤：分步展示解题过程\n"
    "3. 最终答案：给出明确答案"
)


def encode_image(image_or_path):
    """将图像编码为 JPEG bytes"""
    if isinstance(image_or_path, str):
        img = Image.open(image_or_path)
    elif isinstance(image_or_path, Image.Image):
        img = image_or_path
    else:
        raise ValueError(f"不支持的图像类型: {type(image_or_path)}")

    if img.mode in ('RGBA', 'LA', 'P'):
        img = img.convert('RGB')

    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return [buf.getvalue()]  # 保持与现有格式一致（list of bytes）


def convert_scienceqa(output_path, max_samples=None):
    """转换 ScienceQA 数据集"""
    from datasets import load_dataset

    print("下载 ScienceQA 数据集...")
    ds = load_dataset("derek-thomas/ScienceQA", split="train")

    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 ScienceQA"):
        try:
            if item.get('image') is None:
                skipped += 1
                continue

            # 构建问题文本
            question = item.get('question', '')
            choices = item.get('choices', [])
            if choices:
                choice_text = "\n".join(
                    f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
                )
                question = f"{question}\n{choice_text}"

            # 构建答案
            answer_idx = item.get('answer', 0)
            answer_text = f"正确答案是 {chr(65 + answer_idx)}"
            if item.get('lecture'):
                answer_text = f"解析：{item['lecture']}\n\n{answer_text}"

            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer_text},
            ]

            image_bytes = encode_image(item['image'])
            subject = item.get('subject', 'science')
            grade = item.get('grade', '')

            records.append({
                'conversations': json.dumps(conversations, ensure_ascii=False),
                'image_bytes': image_bytes,
            })

            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1

    print(f"ScienceQA: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_mathverse(output_path, max_samples=None):
    """转换 MathVerse 数据集"""
    from datasets import load_dataset

    print("下载 MathVerse 数据集...")
    try:
        # MathVerse 新版本 config 是 'testmini'
        ds = load_dataset("AI4Math/MathVerse", "testmini", split="testmini")
    except Exception as e:
        print(f"MathVerse 数据集加载失败: {e}")
        return

    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 MathVerse"):
        try:
            image = item.get('image')
            if image is None:
                skipped += 1
                continue

            question = item.get('question', item.get('problem', ''))
            answer = item.get('answer', '')
            # MathVerse 没有 solution 字段，answer 已经包含了解析

            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": f"答案：{answer}"},
            ]

            image_bytes = encode_image(image)
            records.append({
                'conversations': json.dumps(conversations, ensure_ascii=False),
                'image_bytes': image_bytes,
            })

            if max_samples and len(records) >= max_samples:
                break
        except Exception as e:
            skipped += 1

    print(f"MathVerse: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_mathvista(output_path, max_samples=None):
    """转换 MathVista 数据集"""
    from datasets import load_dataset

    print("下载 MathVista 数据集...")
    try:
        # MathVista 新版本使用 'default' config, 'testmini' split
        ds = load_dataset("AI4Math/MathVista", "default", split="testmini")
    except Exception as e:
        print(f"MathVista 数据集加载失败: {e}")
        return

    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 MathVista"):
        try:
            # MathVista: decoded_image 是 PIL Image（优先），image 是路径字符串
            image = item.get('decoded_image')
            if image is None:
                skipped += 1
                continue

            question = item.get('question', '')
            choices = item.get('choices')
            if choices and len(choices) > 0:
                question += "\n" + "\n".join(
                    f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)
                )

            answer = item.get('answer', '')
            query_cot = item.get('query', '')

            answer_text = f"答案：{answer}"
            if query_cot:
                answer_text = f"解析：{query_cot}\n\n答案：{answer}"

            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer_text},
            ]

            image_bytes = encode_image(image)
            records.append({
                'conversations': json.dumps(conversations, ensure_ascii=False),
                'image_bytes': image_bytes,
            })

            if max_samples and len(records) >= max_samples:
                break
        except Exception as e:
            skipped += 1

    print(f"MathVista: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_ocr_vqa(output_path, max_samples=None):
    """转换 OCR-VQA 数据集"""
    from datasets import load_dataset

    print("下载 OCR-VQA 数据集...")
    try:
        ds = load_dataset("MMInstruction/OCR-VQA", split="train")
    except Exception as e:
        print(f"OCR-VQA 加载失败: {e}，尝试其他方式...")
        try:
            ds = load_dataset("howard-hou/OCR-VQA", split="train")
        except Exception:
            print("OCR-VQA 所有方式均加载失败，跳过")
            return

    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 OCR-VQA"):
        try:
            image = item.get('image')
            if image is None:
                skipped += 1
                continue

            question = item.get('question', '')
            answer = item.get('answer', item.get('answers', [''])[0])

            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": f"答案：{answer}"},
            ]

            image_bytes = encode_image(image)
            records.append({
                'conversations': json.dumps(conversations, ensure_ascii=False),
                'image_bytes': image_bytes,
            })

            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1

    print(f"OCR-VQA: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def save_parquet(records, output_path):
    """保存为 Parquet 文件"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    conversations_list = [r['conversations'] for r in records]
    image_bytes_list = [r['image_bytes'] for r in records]

    table = pa.table({
        'conversations': pa.array(conversations_list, type=pa.string()),
        'image_bytes': pa.array(image_bytes_list, type=pa.list_(pa.binary())),
    })

    pq.write_table(table, output_path, compression='snappy')
    print(f"已保存 {len(records)} 条记录到: {output_path}")

    # 预览前 2 条
    print("\n预览:")
    for i, conv in enumerate(conversations_list[:2]):
        parsed = json.loads(conv)
        print(f"[{i + 1}] ", end="")
        for turn in parsed:
            content_preview = turn['content'][:60].replace('\n', ' ')
            print(f"{turn['role']}: {content_preview}... | ", end="")
        print()


def convert_ceval(output_path, max_samples=None):
    """转换 C-Eval 数据集（中文52学科，遍历理科子集）"""
    from datasets import load_dataset, get_dataset_config_names
    print("下载 C-Eval 数据集...")
    # 只取理科相关学科
    STEM_CONFIGS = [
        'high_school_mathematics', 'high_school_physics', 'high_school_chemistry',
        'high_school_biology', 'middle_school_mathematics', 'middle_school_physics',
        'middle_school_chemistry', 'middle_school_biology',
        'college_physics', 'college_chemistry', 'college_programming',
        'probability_and_statistics', 'discrete_mathematics', 'advanced_mathematics',
    ]
    records = []
    skipped = 0
    for cfg in STEM_CONFIGS:
        try:
            ds = load_dataset("ceval/ceval-exam", cfg, split="test", streaming=True)
        except Exception:
            continue
        for item in ds:
            try:
                question = item.get('question', '')
                choices = [item.get(k, '') for k in ['A','B','C','D'] if item.get(k)]
                if choices:
                    question += "\n" + "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
                answer = item.get('answer', '')
                explanation = item.get('explanation', '')
                answer_text = f"答案：{answer}"
                if explanation:
                    answer_text = f"解析：{explanation}\n\n{answer_text}"
                conversations = [
                    {"role": "system", "content": "你是一位耐心的辅导老师，请仔细阅读题目，理解题意后给出正确的解答。请包含：1. 题目分析 2. 解题步骤 3. 最终答案"},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer_text},
                ]
                placeholder = Image.new('RGB', (256, 256), (255, 255, 255))
                image_bytes = encode_image(placeholder)
                records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
                if max_samples and len(records) >= max_samples:
                    break
            except Exception:
                skipped += 1
        if max_samples and len(records) >= max_samples:
            break
        print(f"  C-Eval/{cfg}: 累计 {len(records)} 条")
    print(f"C-Eval: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_cmmlu(output_path, max_samples=None):
    """转换 CMMLU 数据集（中文67学科）"""
    from datasets import load_dataset
    print("下载 CMMLU 数据集...")
    try:
        ds = load_dataset("haonan-li/CMMLU", split="train")
    except Exception as e:
        print(f"CMMLU 加载失败: {e}")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 CMMLU"):
        try:
            question = item.get('question', '')
            choices_raw = item.get('choices', [])
            if isinstance(choices_raw, str):
                import ast
                choices_raw = ast.literal_eval(choices_raw)
            if choices_raw:
                question += "\n" + "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices_raw))
            answer = item.get('answer', '')
            answer_text = f"答案：{answer}"
            conversations = [
                {"role": "system", "content": "你是一位耐心的辅导老师，请仔细阅读题目，理解题意后给出正确的解答。请包含：1. 题目分析 2. 解题步骤 3. 最终答案"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer_text},
            ]
            placeholder = Image.new('RGB', (256, 256), (255, 255, 255))
            image_bytes = encode_image(placeholder)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"CMMLU: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_tabmwp(output_path, max_samples=None):
    """转换 TabMWP 数据集（表格数学应用题）"""
    from datasets import load_dataset
    print("下载 TabMWP 数据集...")
    ds = None
    for name in ["lupantech/TabMWP", "ahmed-mahmoud/TabMWP", "lupantech/tabmwp"]:
        try:
            ds = load_dataset(name, split="train", streaming=True)
            break
        except Exception:
            continue
    if ds is None:
        print("TabMWP 加载失败（所有镜像均不可用）")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 TabMWP"):
        try:
            image = item.get('image') or item.get('table_image')
            if image is None:
                skipped += 1
                continue
            question = item.get('question', '')
            answer = item.get('answer', '')
            solution = item.get('solution', '')
            answer_text = answer
            if solution:
                answer_text = f"解析：{solution}\n\n答案：{answer_text}"
            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer_text},
            ]
            image_bytes = encode_image(image)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"TabMWP: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_gaokao(output_path, max_samples=None):
    """转换 GAOKAO-Bench 数据集（高考真题，纯文本）"""
    from datasets import load_dataset
    print("下载 GAOKAO-Bench 数据集...")
    ds = None
    for name in ["OpenLMLab/GAOKAO-Bench", "openlmlab/gaokao-bench"]:
        try:
            ds = load_dataset(name, split="test", streaming=True)
            break
        except Exception:
            continue
    if ds is None:
        print("GAOKAO-Bench 加载失败，跳过")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 GAOKAO-Bench"):
        try:
            question = item.get('question', '')
            answer = item.get('answer', '')
            solution = item.get('solution', item.get('analysis', ''))
            answer_text = answer
            if solution:
                answer_text = f"解析：{solution}\n\n答案：{answer_text}"
            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer_text},
            ]
            placeholder = Image.new('RGB', (256, 256), (255, 255, 255))
            image_bytes = encode_image(placeholder)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"GAOKAO-Bench: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_math23k(output_path, max_samples=None):
    """转换 Math23K 数据集（中文小学数学应用题）"""
    from datasets import load_dataset
    print("下载 Math23K 数据集...")
    ds = None
    for name in ["math23k/math23k", "EvanWang/math23k", "Sumbee/math23k"]:
        try:
            ds = load_dataset(name, split="train", streaming=True)
            break
        except Exception:
            continue
    if ds is None:
        print("Math23K 加载失败，跳过")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 Math23K"):
        try:
            question = item.get('question', item.get('text', ''))
            answer = item.get('answer', item.get('equation', ''))
            answer_text = f"答案：{answer}"
            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer_text},
            ]
            placeholder = Image.new('RGB', (256, 256), (255, 255, 255))
            image_bytes = encode_image(placeholder)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"Math23K: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_ape210k(output_path, max_samples=None):
    """转换 Ape210K 数据集（中文小学数学，最大规模）"""
    from datasets import load_dataset
    print("下载 Ape210K 数据集...")
    ds = None
    # 新 HF 路径：MU-NLPC/Calc-ape210k
    for name in ["MU-NLPC/Calc-ape210k", "Ape210K/Ape210K", "Chenny0808/ape210k", "EvanWang/ape210k"]:
        try:
            ds = load_dataset(name, split="train", streaming=True)
            break
        except Exception:
            continue
    if ds is None:
        print("Ape210K 加载失败，跳过")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 Ape210K"):
        try:
            # 兼容多种字段名
            question = item.get('question_chinese') or item.get('question') or item.get('text', '')
            answer = str(item.get('result') or item.get('answer') or item.get('equation', ''))
            solution = item.get('chain', '')
            answer_text = f"答案：{answer}"
            if solution:
                answer_text = f"解析：{solution}\n\n{answer_text}"
            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer_text},
            ]
            placeholder = Image.new('RGB', (256, 256), (255, 255, 255))
            image_bytes = encode_image(placeholder)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"Ape210K: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


# 数据集转换函数映射


def convert_mmmu(output_path, max_samples=None):
    """转换 MMMU 数据集（30学科多模态大学级图文题，含中文图表）"""
    from datasets import load_dataset, get_dataset_config_names
    print("下载 MMMU 数据集...")
    try:
        configs = get_dataset_config_names("MMMU/MMMU")
    except Exception:
        configs = []
    if not configs:
        print("MMMU 加载失败: 无法获取 config 列表")
        return
    print(f"  MMMU 共 {len(configs)} 个学科 config，逐个处理...")
    records = []
    skipped = 0
    for cfg in configs:
        try:
            ds = load_dataset("MMMU/MMMU", cfg, split="test", streaming=True)
        except Exception:
            continue
        for item in ds:
            try:
                question = item.get('question', '')
                options = item.get('options', item.get('choices', ''))
                if isinstance(options, list):
                    question += "\n" + "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
                elif isinstance(options, str) and options:
                    question += "\n" + options
                answer = str(item.get('answer', item.get('gt', '')))
                explanation = item.get('explanation', item.get('rationale', ''))
                answer_text = f"答案：{answer}"
                if explanation:
                    answer_text = f"解析：{explanation}\n\n{answer_text}"
                image = item.get('image_1') or item.get('image') or item.get('images')
                if isinstance(image, list) and len(image) > 0:
                    image = image[0]
                if image is None:
                    placeholder = Image.new('RGB', (256, 256), (255, 255, 255))
                    image = placeholder
                conversations = [
                    {"role": "system", "content": EDU_SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>\n{question}"},
                    {"role": "assistant", "content": answer_text},
                ]
                image_bytes = encode_image(image)
                records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
                if max_samples and len(records) >= max_samples:
                    break
            except Exception:
                skipped += 1
            if max_samples and len(records) >= max_samples:
                break
        if max_samples and len(records) >= max_samples:
            break
        print(f"  MMMU/{cfg}: 累计 {len(records)} 条")
    print(f"MMMU: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_geoqa(output_path, max_samples=None):
    """转换 GeoQA 数据集（中文几何选择题）"""
    from datasets import load_dataset
    print("下载 GeoQA 数据集...")
    ds = None
    for name in ["leonardPKU/GEOQA_R1V_Train_8K", "GeoQA/GeoQA-Plus", "AIM3/GeoQA"]:
        try:
            ds = load_dataset(name, split="train", streaming=True)
            break
        except Exception:
            continue
    if ds is None:
        print("GeoQA 加载失败，跳过")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 GeoQA"):
        try:
            image = item.get('image') or item.get('img')
            if image is None:
                skipped += 1
                continue
            # 兼容多种字段名
            question = item.get('problem') or item.get('question', item.get('text', ''))
            answer = str(item.get('solution') or item.get('answer', item.get('gt', '')))
            answer_text = f"答案：{answer}"
            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer_text},
            ]
            image_bytes = encode_image(image)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"GeoQA: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_biology(output_path, max_samples=None):
    """转换生物相关数据集（BioVQA + VQA-RAD）"""
    from datasets import load_dataset
    print("下载 BioVQA 数据集...")
    records = []
    skipped = 0
    # BioVQA
    try:
        ds = load_dataset("MMInstruction/BioVQA", split="train", streaming=True)
    except Exception:
        ds = None
    if ds:
        for item in tqdm(ds, desc="处理 BioVQA"):
            try:
                image = item.get('image')
                if image is None:
                    skipped += 1
                    continue
                question = item.get('question', '')
                answer = str(item.get('answer', ''))
                conversations = [
                    {"role": "system", "content": EDU_SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>\n{question}"},
                    {"role": "assistant", "content": f"答案：{answer}"},
                ]
                image_bytes = encode_image(image)
                records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
                if max_samples and len(records) >= max_samples:
                    break
            except Exception:
                skipped += 1
    print(f"BioVQA: 累计 {len(records)} 条")
    save_parquet(records, output_path)


# ============================
# 新增数据集转换函数
# ============================


def convert_geometry3k(output_path, max_samples=None):
    """转换 Geometry3K 数据集（几何图文题，含详细解析）"""
    from datasets import load_dataset
    print("下载 Geometry3K 数据集...")
    ds = None
    for name in ["hiyouga/geometry3k", "MMInstruction/Geometry3K"]:
        try:
            ds = load_dataset(name, split="train", streaming=True)
            break
        except Exception:
            continue
    if ds is None:
        print("Geometry3K 所有镜像加载失败，跳过")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 Geometry3K"):
        try:
            # 兼容多种字段名
            image = item.get('images') or item.get('image')
            if image is None:
                skipped += 1
                continue
            question = item.get('problem') or item.get('question', '')
            answer = item.get('answer', '')
            solution = item.get('solution', item.get('explanation', ''))
            answer_text = f"答案：{answer}"
            if solution:
                answer_text = f"解析：{solution}\n\n{answer_text}"
            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer_text},
            ]
            image_bytes = encode_image(image)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"Geometry3K: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_chartqa(output_path, max_samples=None):
    """转换 ChartQA 数据集（图表问答，柱状图/折线图/饼图）"""
    from datasets import load_dataset
    print("下载 ChartQA 数据集...")
    records = []
    skipped = 0
    for name in ["MMInstruction/ChartQA", "iamshnoo/chartqa", "HuggingFaceM4/ChartQA"]:
        try:
            ds = load_dataset(name, split="train", streaming=True)
            break
        except Exception:
            ds = None
    if ds is None:
        print("ChartQA 所有镜像加载失败，跳过")
        return
    for item in tqdm(ds, desc="处理 ChartQA"):
        try:
            image = item.get('image') or item.get('img')
            if image is None:
                skipped += 1
                continue
            question = item.get('question', item.get('query', ''))
            answer = item.get('answer', item.get('label', ''))
            if isinstance(answer, list):
                answer = ', '.join(str(a) for a in answer)
            answer_text = f"答案：{answer}"
            conversations = [
                {"role": "system", "content": "你是一位数据解读老师，请根据图表信息回答问题。"},
                {"role": "user", "content": f"<image>\n请根据图表回答：{question}"},
                {"role": "assistant", "content": answer_text},
            ]
            image_bytes = encode_image(image)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"ChartQA: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_dvqa(output_path, max_samples=None):
    """转换 DVQA 数据集（信息图/图表问答）"""
    from datasets import load_dataset
    print("下载 DVQA 数据集...")
    ds = None
    for name in ["DavidNguyen/DVQA", "MMInstruction/DVQA"]:
        try:
            ds = load_dataset(name, split="train", streaming=True)
            break
        except Exception:
            continue
    if ds is None:
        print("DVQA 所有镜像加载失败，跳过")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 DVQA"):
        try:
            # 兼容 DavidNguyen/DVQA 的 png 字段
            image = item.get('image') or item.get('png')
            if image is None:
                skipped += 1
                continue
            question = item.get('question', '')
            answer = str(item.get('answer', ''))
            answer_text = f"答案：{answer}"
            conversations = [
                {"role": "system", "content": "你是一位信息解读老师，请根据信息图回答问题。"},
                {"role": "user", "content": f"<image>\n请根据信息图回答：{question}"},
                {"role": "assistant", "content": answer_text},
            ]
            image_bytes = encode_image(image)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"DVQA: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_ai2d(output_path, max_samples=None):
    """转换 AI2D 数据集（科学示意图，适合亲子科学教育）"""
    from datasets import load_dataset
    print("下载 AI2D 数据集...")
    records = []
    skipped = 0
    for name in ["MMInstruction/AI2D", "allenai/AI2D", "alkampfermit/AI2D"]:
        try:
            ds = load_dataset(name, split="train", streaming=True)
            break
        except Exception:
            ds = None
    if ds is None:
        print("AI2D 所有镜像加载失败，跳过")
        return
    for item in tqdm(ds, desc="处理 AI2D"):
        try:
            image = item.get('image')
            if image is None:
                skipped += 1
                continue
            question = item.get('question', '')
            answer = str(item.get('answer', ''))
            answer_text = f"答案：{answer}"
            conversations = [
                {"role": "system", "content": "你是一位科学老师，请观察示意图帮助孩子理解科学概念。"},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer_text},
            ]
            image_bytes = encode_image(image)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"AI2D: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_vizwiz(output_path, max_samples=None):
    """转换 VizWiz 数据集（真实场景模糊/倾斜图像，提高模型鲁棒性）"""
    from datasets import load_dataset
    print("下载 VizWiz 数据集...")
    try:
        ds = load_dataset("MMInstruction/VizWiz", split="train", streaming=True)
    except Exception as e:
        print(f"VizWiz 加载失败: {e}")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 VizWiz"):
        try:
            image = item.get('image')
            if image is None:
                skipped += 1
                continue
            question = item.get('question', '')
            answer = item.get('answer', '')
            if isinstance(answer, list):
                answer = '; '.join(str(a) for a in answer)
            answer_text = f"答案：{answer}"
            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer_text},
            ]
            image_bytes = encode_image(image)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"VizWiz: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_tqa(output_path, max_samples=None):
    """转换 TQA (TextbookQA) 数据集（教科书级别图文问答）"""
    from datasets import load_dataset
    print("下载 TQA 数据集...")
    try:
        ds = load_dataset("MMInstruction/TQA", split="train", streaming=True)
    except Exception as e:
        print(f"TQA 加载失败: {e}")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 TQA"):
        try:
            image = item.get('image')
            if image is None:
                skipped += 1
                continue
            question = item.get('question', '')
            choices = item.get('choices', [])
            if choices:
                question += "\n" + "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
            answer = str(item.get('answer', ''))
            explanation = item.get('explanation', item.get('rationale', ''))
            answer_text = f"答案：{answer}"
            if explanation:
                answer_text = f"解析：{explanation}\n\n{answer_text}"
            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer_text},
            ]
            image_bytes = encode_image(image)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"TQA: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_clevr_math(output_path, max_samples=None):
    """转换 CLEVR-Math 数据集（合成图数学推理，增强空间/几何推理）"""
    from datasets import load_dataset
    print("下载 CLEVR-Math 数据集...")
    records = []
    skipped = 0
    for name in ["MMInstruction/CLEVR-Math", "dali-does/clevr-math"]:
        try:
            ds = load_dataset(name, split="train", streaming=True)
            break
        except Exception:
            ds = None
    if ds is None:
        print("CLEVR-Math 所有镜像加载失败，跳过")
        return
    for item in tqdm(ds, desc="处理 CLEVR-Math"):
        try:
            image = item.get('image')
            if image is None:
                skipped += 1
                continue
            question = item.get('question', '')
            answer = str(item.get('answer', ''))
            program = item.get('program', item.get('program_text', ''))
            answer_text = f"答案：{answer}"
            if program:
                answer_text = f"推理过程：{program}\n\n{answer_text}"
            conversations = [
                {"role": "system", "content": "你是一位数学思维训练老师，请观察图像中的几何/空间关系，解答问题。"},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer_text},
            ]
            image_bytes = encode_image(image)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"CLEVR-Math: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_race(output_path, max_samples=None):
    """转换 RACE 数据集（中英文阅读理解，纯文本+占位图）"""
    from datasets import load_dataset
    print("下载 RACE 数据集...")
    records = []
    skipped = 0
    for name in ["hfl/race", "race/race"]:
        try:
            ds = load_dataset(name, "middle", split="train", streaming=True)
            break
        except Exception:
            ds = None
    if ds is None:
        try:
            ds = load_dataset("ehovy/race", "all", split="train", streaming=True)
        except Exception:
            print("RACE 加载失败，跳过")
            return
    for item in tqdm(ds, desc="处理 RACE"):
        try:
            article = item.get('article', '')
            question = item.get('question', '')
            options = item.get('options', [])
            answer = str(item.get('answer', ''))
            if options:
                question = f"阅读以下文章，回答问题：\n\n文章：{article[:200]}...\n\n{question}\n" + "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(options))
            else:
                question = f"阅读以下文章，回答问题：\n\n文章：{article[:200]}...\n\n{question}"
            answer_text = f"答案：{answer}"
            placeholder = Image.new('RGB', (256, 256), (255, 255, 255))
            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer_text},
            ]
            image_bytes = encode_image(placeholder)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"RACE: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_openr1_math(output_path, max_samples=None):
    """转换 OpenR1-Math CN K12 数据集（91K 中文数学推理链）"""
    from datasets import load_dataset
    print("下载 OpenR1-Math CN K12 数据集...")
    try:
        ds = load_dataset("Neelectric/OpenR1-Math-cn_k12-91k", split="train", streaming=True)
    except Exception as e:
        print(f"OpenR1-Math 加载失败: {e}")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 OpenR1-Math"):
        try:
            question = item.get('problem', '')
            solution = item.get('solution', '')
            answer = item.get('answer', '')
            answer_text = f"答案：{answer}"
            if solution:
                answer_text = f"解析：{solution}\n\n{answer_text}"
            conversations = [
                {"role": "system", "content": "你是一位数学辅导老师，请仔细阅读题目，分步解答。请包含：1. 题目分析 2. 解题步骤 3. 最终答案"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer_text},
            ]
            placeholder = Image.new('RGB', (256, 256), (255, 255, 255))
            image_bytes = encode_image(placeholder)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"OpenR1-Math: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_gaokao_mathqa(output_path, max_samples=None):
    """转换 AGIEval Gaokao MathQA 数据集（高考数学选择题）"""
    from datasets import load_dataset
    print("下载 Gaokao MathQA 数据集...")
    try:
        ds = load_dataset("hails/agieval-gaokao-mathqa", split="test", streaming=True)
    except Exception as e:
        print(f"Gaokao MathQA 加载失败: {e}")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 Gaokao MathQA"):
        try:
            query = item.get('query', '').replace('问题：', '').strip()
            choices = item.get('choices', [])
            gold = item.get('gold', [])
            answer_idx = gold[0] if isinstance(gold, list) and len(gold) > 0 else gold
            if isinstance(answer_idx, int):
                answer_letter = chr(65 + answer_idx)
            else:
                answer_letter = str(answer_idx)
            question = query
            if choices:
                question += "\n" + "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
            answer_text = f"答案：{answer_letter}"
            conversations = [
                {"role": "system", "content": "你是一位数学辅导老师，请仔细阅读题目，分步解答。请包含：1. 题目分析 2. 解题步骤 3. 最终答案"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer_text},
            ]
            placeholder = Image.new('RGB', (256, 256), (255, 255, 255))
            image_bytes = encode_image(placeholder)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"Gaokao MathQA: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_gaokao_mathcloze(output_path, max_samples=None):
    """转换 AGIEval Gaokao MathCloze 数据集（高考数学填空题）"""
    from datasets import load_dataset
    print("下载 Gaokao MathCloze 数据集...")
    try:
        ds = load_dataset("hails/agieval-gaokao-mathcloze", split="test", streaming=True)
    except Exception as e:
        print(f"Gaokao MathCloze 加载失败: {e}")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 Gaokao MathCloze"):
        try:
            query = item.get('query', '').replace('问题：', '').strip()
            answer = str(item.get('answer', ''))
            answer_text = f"答案：{answer}"
            conversations = [
                {"role": "system", "content": "你是一位数学辅导老师，请仔细阅读题目，分步解答。请包含：1. 题目分析 2. 解题步骤 3. 最终答案"},
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer_text},
            ]
            placeholder = Image.new('RGB', (256, 256), (255, 255, 255))
            image_bytes = encode_image(placeholder)
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"Gaokao MathCloze: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_we_math(output_path, max_samples=None):
    """转换 We-Math 2.0 数据集（系统性数学知识体系）"""
    from datasets import load_dataset
    print("下载 We-Math 2.0 数据集...")
    try:
        ds = load_dataset("We-Math/We-Math2.0-Standard", split="train", streaming=True)
    except Exception as e:
        print(f"We-Math 2.0 加载失败: {e}")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 We-Math 2.0"):
        try:
            # 提取问题文本
            question = item.get('problem', item.get('question', ''))
            answer = item.get('answer', item.get('solution', ''))
            
            # 构建对话
            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"答案：{answer}"},
            ]
            
            # 处理图像
            image = item.get('image') or item.get('img')
            if image:
                image_bytes = encode_image(image)
            else:
                placeholder = Image.new('RGB', (256, 256), (255, 255, 255))
                image_bytes = encode_image(placeholder)
            
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"We-Math 2.0: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_geo170k(output_path, max_samples=None):
    """转换 Geo170K 数据集（几何推理专项）"""
    from datasets import load_dataset
    print("下载 Geo170K 数据集...")
    try:
        ds = load_dataset("Luckyjhg/Geo170K", split="train", streaming=True)
    except Exception as e:
        print(f"Geo170K 加载失败: {e}")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 Geo170K"):
        try:
            # 提取问题和答案
            question = item.get('question', item.get('problem', ''))
            answer = str(item.get('answer', item.get('solution', '')))
            
            # 构建对话
            conversations = [
                {"role": "system", "content": "你是一位几何推理老师，请仔细观察图形，理解问题并给出解答。"},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": f"答案：{answer}"},
            ]
            
            # 处理图像
            image = item.get('image') or item.get('img')
            if image:
                image_bytes = encode_image(image)
            else:
                placeholder = Image.new('RGB', (256, 256), (255, 255, 255))
                image_bytes = encode_image(placeholder)
            
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"Geo170K: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_cmm_math(output_path, max_samples=None):
    """转换 CMM-Math 数据集（中文K12数学图文）"""
    from datasets import load_dataset
    print("下载 CMM-Math 数据集...")
    try:
        # 尝试从 GitHub 或 HuggingFace 加载
        ds = load_dataset("ECNU-ICALK/EduChat-Math", "CMM-Math", split="train", streaming=True)
    except Exception as e1:
        try:
            ds = load_dataset("wangronglu/CEIM-Dataset", split="train", streaming=True)
        except Exception as e2:
            print(f"CMM-Math 加载失败: {e1}, {e2}")
            return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 CMM-Math"):
        try:
            # 提取问题和答案
            question = item.get('question', item.get('problem', ''))
            answer = item.get('answer', item.get('solution', ''))
            
            # 构建对话
            conversations = [
                {"role": "system", "content": EDU_SYSTEM_PROMPT},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": f"答案：{answer}"},
            ]
            
            # 处理图像
            image = item.get('image') or item.get('img')
            if image:
                image_bytes = encode_image(image)
            else:
                placeholder = Image.new('RGB', (256, 256), (255, 255, 255))
                image_bytes = encode_image(placeholder)
            
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"CMM-Math: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_math_real(output_path, max_samples=None):
    """转换 MathReal 数据集（真实场景K12数学）"""
    from datasets import load_dataset
    print("下载 MathReal 数据集...")
    try:
        # 从 HuggingFace 加载
        ds = load_dataset("MathReal/MathReal", split="train", streaming=True)
    except Exception as e:
        print(f"MathReal 加载失败: {e}")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 MathReal"):
        try:
            # 提取问题和答案
            question = item.get('question', item.get('problem', ''))
            answer = str(item.get('answer', item.get('solution', '')))
            
            # 构建对话
            conversations = [
                {"role": "system", "content": "你是一位数学老师，请仔细观察图片，理解真实场景中的数学问题并给出解答。"},
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": f"答案：{answer}"},
            ]
            
            # 处理图像
            image = item.get('image') or item.get('img')
            if image:
                image_bytes = encode_image(image)
            else:
                placeholder = Image.new('RGB', (256, 256), (255, 255, 255))
                image_bytes = encode_image(placeholder)
            
            records.append({'conversations': json.dumps(conversations, ensure_ascii=False), 'image_bytes': image_bytes})
            if max_samples and len(records) >= max_samples:
                break
        except Exception:
            skipped += 1
    print(f"MathReal: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


CONVERTERS = {
    'scienceqa': convert_scienceqa,
    'mathverse': convert_mathverse,
    'mathvista': convert_mathvista,
    'ocr_vqa': convert_ocr_vqa,
    'ceval': convert_ceval,
    'cmmlu': convert_cmmlu,
    'tabmwp': convert_tabmwp,
    'gaokao': convert_gaokao,
    'math23k': convert_math23k,
    'ape210k': convert_ape210k,
    'mmmu': convert_mmmu,
    'geoqa': convert_geoqa,
    'biology': convert_biology,
    'geometry3k': convert_geometry3k,
    'chartqa': convert_chartqa,
    'dvqa': convert_dvqa,
    'ai2d': convert_ai2d,
    'vizwiz': convert_vizwiz,
    'tqa': convert_tqa,
    'clevr_math': convert_clevr_math,
    'race': convert_race,
    'openr1_math': convert_openr1_math,
    'gaokao_mathqa': convert_gaokao_mathqa,
    'gaokao_mathcloze': convert_gaokao_mathcloze,
    # 新增数据集
    'we_math': convert_we_math,
    'geo170k': convert_geo170k,
    'cmm_math': convert_cmm_math,
    'math_real': convert_math_real,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="教育数据集格式转换")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(CONVERTERS.keys()),
                        help="数据集名称")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 Parquet 路径")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大转换样本数（调试用）")
    args = parser.parse_args()

    converter = CONVERTERS[args.dataset]
    converter(args.output, args.max_samples)