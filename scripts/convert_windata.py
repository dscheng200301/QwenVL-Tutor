"""
转换 windata-vision-synthetics-zh-300k 数据集为 QwenSearch Parquet 格式
支持中文多模态图文数据，包含文档、图表、数学等多种场景

使用方法:
    python scripts/convert_windata.py --input dataset/windata-300k --output dataset/edu_windata.parquet
"""

import os
import sys
import json
import argparse
import io
from pathlib import Path
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


def load_image(image_path):
    """加载图像"""
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except Exception as e:
        print(f"加载图像失败 {image_path}: {e}")
        return None


def convert_conversation(conv_data):
    """转换对话格式为 QwenSearch 格式"""
    conversations = []
    
    for turn in conv_data:
        role = 'user' if turn['from'] == 'human' else 'assistant'
        content = turn['value']
        
        # 处理 <image> 标记
        if '<image>' in content:
            # 保持 <image> 标记，edu_dataset 会自动处理
            conversations.append({
                'role': role,
                'content': content
            })
        else:
            conversations.append({
                'role': role,
                'content': content
            })
    
    return conversations


def extract_images_from_conversations(conv_data, base_path):
    """从对话中提取图像文件路径"""
    images = []
    for turn in conv_data:
        content = turn['value']
        # 图像路径通常在 conversations 外的 image 字段
    return images


def convert_windata_dataset(input_path, output_path, max_samples=None, sample_rate=1.0):
    """
    转换 windata 数据集为 Parquet 格式
    
    Args:
        input_path: windata 数据集路径
        output_path: 输出 Parquet 文件路径
        max_samples: 最大采样数量（None 表示全部）
        sample_rate: 采样率（0.0-1.0）
    """
    # 图像路径映射（JSON中的名称 -> 实际目录）
    IMAGE_PATH_MAP = {
        'Docmatix': 'Docmatix',  # 需要从 HuggingFace 下载
        'Tallyqa': 'Tallyqa',    # 需要从源数据下载
        'PlotQA': 'PlotQA',      # 需要从 PlotQA.tar.gz 解压
        'geo170k': 'data/local/Open-source-data/geo170k',  # 已解压
        'EST_VQA': 'EST_VQA',    # 需要从源数据下载
        'arxivqa': 'arxivqa',    # 需要从 HuggingFace 下载
    }
    
    print(f"开始转换 windata 数据集...")
    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")
    
    # JSON 文件列表（排除非数据文件）
    # 包含所有数据集，包括数学类 geo170k
    json_files = [
        # 'Docmatix-synthetic.json',      # 文档类（文件大，跳过）
        # 'Tallyqa-synthetic.json',       # 通用文档（文件太大，内存溢出）
        # 'PlotQA-synthetic.json',        # 图表类
        'geo170k-synthetic.json',       # 数学类（已解压图像，重点保留）
        # 'EST_VQA-synthetic.json',       # OCR类
        # 'arxivqa-synthetic.json',       # 论文截图
    ]
    
    records = []
    total_processed = 0
    
    for json_file in json_files:
        json_path = os.path.join(input_path, json_file)
        if not os.path.exists(json_path):
            print(f"文件不存在，跳过: {json_file}")
            continue
        
        print(f"\n处理文件: {json_file}")
        
        # 流式读取 JSON 文件（避免内存溢出）
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"  原始数据量: {len(data)}")
        
        # 采样
        if sample_rate < 1.0:
            import random
            sample_size = int(len(data) * sample_rate)
            data = random.sample(data, sample_size)
            print(f"  采样后数量: {len(data)}")
        
        # 限制最大数量
        if max_samples and total_processed >= max_samples:
            remaining = max_samples - (total_processed - len(data))
            if remaining <= 0:
                break
            data = data[:remaining]
        
        # 提取图像归档
        tar_file = None
        if json_file == 'geo170k-synthetic.json':
            tar_file = os.path.join(input_path, 'geo170k.tar.gz')
        
        for item in tqdm(data, desc=f"  转换 {json_file}"):
            try:
                # 转换对话格式
                conversations = convert_conversation(item['conversations'])
                
                # 获取图像路径
                image_files = item.get('image', [])
                image = None
                
                if image_files:
                    image_folder_key = item.get('image_file', '')
                    image_name = image_files[0]
                    
                    # 映射到实际目录
                    actual_folder = IMAGE_PATH_MAP.get(image_folder_key, image_folder_key)
                    image_path = os.path.join(input_path, actual_folder, image_name)
                    
                    if os.path.exists(image_path):
                        image = load_image(image_path)
                
                # 编码图像为 bytes
                if image:
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='JPEG', quality=85)
                    image_bytes = [img_buffer.getvalue()]
                else:
                    # 无图像数据，跳过
                    continue
                
                # 构建记录
                record = {
                    'conversations': json.dumps(conversations, ensure_ascii=False),
                    'image_bytes': image_bytes,
                    'subject': 'math' if 'math' in json_file.lower() or 'geo' in json_file.lower() else 'general',
                    'grade_level': 'K12',
                    'question_type': 'synthetic',
                    'source': json_file.replace('-synthetic.json', ''),
                }
                
                records.append(record)
                total_processed += 1
                
                # 检查是否达到最大数量
                if max_samples and total_processed >= max_samples:
                    break
                    
            except Exception as e:
                print(f"  处理记录失败: {e}")
                continue
        
        if max_samples and total_processed >= max_samples:
            break
    
    print(f"\n转换完成，总记录数: {len(records)}")
    
    # 保存为 Parquet
    df = pa.Table.from_pylist(records)
    pq.write_table(df, output_path)
    print(f"已保存到: {output_path}")
    
    return len(records)


def main():
    parser = argparse.ArgumentParser(description="转换 windata 数据集为 Parquet 格式")
    parser.add_argument("--input", type=str, default="dataset/windata-300k", help="输入路径")
    parser.add_argument("--output", type=str, default="dataset/edu_windata.parquet", help="输出路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最大采样数量")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="采样率 (0.0-1.0)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # 执行转换
    convert_windata_dataset(
        args.input,
        args.output,
        args.max_samples,
        args.sample_rate
    )


if __name__ == "__main__":
    main()