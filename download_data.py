"""一键下载 QwenSearch P0 核心数据集"""
import os
import sys
import subprocess

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# 要下载的数据集列表（名称，输出文件名，样本限制）
DATASETS = [
    # === 英文理科（含图）===
    ("scienceqa", "dataset/edu_science.parquet", None),       # 全量 21K
    ("mathverse", "dataset/edu_math_verse.parquet", None),    # 3.9K
    ("mathvista", "dataset/edu_math_vista.parquet", None),    # 1K
    ("ocr_vqa", "dataset/edu_ocr.parquet", 20000),            # 取前 20K
    ("tabmwp", "dataset/edu_tabmwp.parquet", None),           # 表格数学 38K
    # === 中文理科 ===
    ("ceval", "dataset/edu_ceval.parquet", None),             # C-Eval 52学科 14K
    ("cmmlu", "dataset/edu_cmmlu.parquet", None),             # CMMLU 67学科 11K
    ("gaokao", "dataset/edu_gaokao.parquet", None),           # 高考真题
    ("math23k", "dataset/edu_math23k.parquet", 20000),        # 中文小学数学（取20K）
    ("ape210k", "dataset/edu_ape210k.parquet", 20000),        # 中文小学数学（取20K）
    # === 补充：多模态理科 ===
    ("mmmu", "dataset/edu_mmmu.parquet", 5000),               # 大学级图文题（取5K）
    ("geoqa", "dataset/edu_geoqa.parquet", 10000),            # 中文几何（取10K）
    ("biology", "dataset/edu_biology.parquet", 5000),         # 生物图文题 5K
    # ====== 新增：图文数学 ======
    ("geometry3k", "dataset/edu_geometry3k.parquet", 3000),   # 几何图文 3K（全量）
    ("clevr_math", "dataset/edu_clevr_math.parquet", 5000),   # 合成图数学推理（取5K）
    # ====== 新增：图表理解 ======
    ("chartqa", "dataset/edu_chartqa.parquet", 10000),        # 图表题（取10K）
    ("dvqa", "dataset/edu_dvqa.parquet", 5000),              # 信息图问答（取5K）
    # ====== 新增：科学常识 ======
    ("ai2d", "dataset/edu_ai2d.parquet", 4000),               # 科学示意图 5K（取4K）
    ("tqa", "dataset/edu_tqa.parquet", 5000),                 # 教科书图文（取5K）
    # ====== 新增：场景/文本 ======
    ("vizwiz", "dataset/edu_vizwiz.parquet", 5000),           # 真实模糊场景（取5K）
    ("race", "dataset/edu_race.parquet", 10000),              # 阅读理解（取10K）
]

def download_one(dataset_name, output_path, max_samples):
    """下载一个数据集"""
    print(f"\n{'='*60}")
    print(f"📥 正在下载: {dataset_name} -> {output_path}")
    print(f"{'='*60}")
    
    if os.path.exists(output_path):
        print(f"✅ 文件已存在，跳过: {output_path}")
        return True
    
    cmd = [
        sys.executable,
        "scripts/convert_edu_data.py",
        "--dataset", dataset_name,
        "--output", output_path,
    ]
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])
    
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0

if __name__ == "__main__":
    print("🚀 QwenSearch 数据集一键下载")
    print(f"共 {len(DATASETS)} 个数据集待下载\n")
    
    success = 0
    failed = []
    
    for name, path, max_samples in DATASETS:
        if download_one(name, path, max_samples):
            success += 1
        else:
            failed.append(name)
    
    print(f"\n{'='*60}")
    print(f"📊 下载完成: 成功 {success}/{len(DATASETS)}")
    if failed:
        print(f"⚠️ 失败: {', '.join(failed)}")
    else:
        print("🎉 全部下载成功！")
    print(f"{'='*60}")