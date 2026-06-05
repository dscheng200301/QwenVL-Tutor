"""
教育数据集格式转换工具
将多个开源教育数据集统一转换为 QwenSearch 的 Parquet 格式

支持的数据集:
    - scienceqa: ScienceQA (HuggingFace: derek-thomas/ScienceQA)
    - mathverse: MathVerse (HuggingFace: AI4Math/MathVerse)
    - mathvista: MathVista (HuggingFace: AI4Math/MathVista)
    - ocr_vqa: OCR-VQA (HuggingFace: MMInstruction/OCR-VQA)
    - tabmwp: TabMWP (HuggingFace: lupantech/TabMWP)
    - ceval: C-Eval (HuggingFace: ceval/ceval-exam) - 纯文本需截图
    - geoqa_plus: GeoQA+ (HuggingFace: GeoQA/GeoQA-Plus)
    - tqa: TQA TextbookQA (HuggingFace: MMInstruction/TQA)
    - mmmu: MMMU (HuggingFace: MMMU/MMMU)

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
    for name in ["Ape210K/Ape210K", "Chenny0808/ape210k", "EvanWang/ape210k"]:
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
    print(f"Ape210K: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


# 数据集转换函数映射


def convert_mmmu(output_path, max_samples=None):
    """转换 MMMU 数据集（30学科多模态大学级图文题，含中文图表）"""
    from datasets import load_dataset
    print("下载 MMMU 数据集...")
    try:
        ds = load_dataset("MMMU/MMMU", split="test", streaming=True)
    except Exception as e:
        print(f"MMMU 加载失败: {e}")
        return
    records = []
    skipped = 0
    for item in tqdm(ds, desc="处理 MMMU"):
        try:
            # MMMU 字段因学科而异，尽量兼容
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
            # MMMU 图像字段可能是 image_1, image_2, image, images 等
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
    print(f"MMMU: 成功 {len(records)} 条, 跳过 {skipped} 条")
    save_parquet(records, output_path)


def convert_geoqa(output_path, max_samples=None):
    """转换 GeoQA 数据集（中文几何选择题）"""
    from datasets import load_dataset
    print("下载 GeoQA 数据集...")
    ds = None
    for name in ["GeoQA/GeoQA-Plus", "AIM3/GeoQA"]:
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
            question = item.get('question', item.get('text', ''))
            choices = item.get('choices', item.get('options', []))
            if isinstance(choices, list) and choices:
                question += "\n" + "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
            answer = str(item.get('answer', item.get('gt', '')))
            explanation = item.get('explanation', item.get('solution', ''))
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