# QwenSearch — 亲子教育 VLM

<div align="center">

**📸 拍题即答 · 分步引导 · 亲子共学**

基于 Qwen2-VL 基座魔改的拍照做题 VLM，专为亲子教育场景设计。

</div>

---

## 🏗️ 训练管线总览

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   SFT 阶段    │ →  │   DPO 阶段    │ →  │   GRPO 阶段   │
│   监督微调    │    │   偏好对齐    │    │   强化优化    │
├──────────────┤    ├──────────────┤    ├──────────────┤
│ 使用数据:     │    │ 使用数据:     │    │ 使用数据:     │
│ 6个数据集     │    │ ScienceQA    │    │ ScienceQA    │
│ 45,729 条    │    │  6,218 条    │    │  6,218 条    │
├──────────────┤    ├──────────────┤    ├──────────────┤
│ Dataset:      │    │ Dataset:      │    │ Dataset:      │
│ EduDataset    │    │ EduDPODataset │    │ EduGRPODataset│
├──────────────┤    ├──────────────┤    ├──────────────┤
│ Loss:         │    │ Loss:         │    │ Loss:         │
│ CrossEntropy  │    │ DPO loss      │    │ PPO-clip      │
│               │    │ (log-sigmoid) │    │               │
├──────────────┤    ├──────────────┤    ├──────────────┤
│ 评估:         │    │ 评估:         │    │ 评估:         │
│ --stage sft   │    │ --stage dpo   │    │ --stage grpo  │
│ ScienceQA(500)│    │ Skill保持(50) │    │ Skill保持(50) │
│ C-Eval(200)   │    │ 偏好gap(100)  │    │ 奖励质量(100) │
│ Custom(200)   │    │ 退化检测(50)  │    │ 退化检测(50)  │
├──────────────┤    ├──────────────┤    ├──────────────┤
│ 目标指标:     │    │ 目标指标:     │    │ 目标指标:     │
│ 准确率>50%    │    │ gap>0.05      │    │ avg_reward    │
│ 步骤率>60%    │    │ 准确率不掉    │    │   >0.5       │
└──────────────┘    └──────────────┘    └──────────────┘
         ↓                   ↓                  ↓
    保存: edu_sft         保存: edu_dpo        保存: edu_grpo
         ↓                                       ↓
    ┌─────────────────────────────────────────────┐
    │          最终发布评估: --stage full          │
    │  ScienceQA test holdout (4,241)             │
    │  C-Eval holdout (500)                       │
    └─────────────────────────────────────────────┘
```

## 📊 数据集总览（23个，~200K+ 条）

| 类别 | 数据集 | 语言 | 条数 | 含图 | 用途 | 阶段 |
|------|--------|:----:|:----:|:----:|------|:----:|
| **核心图文数学** | ScienceQA | EN | 21,000 | ✅ | 全理科图文，带 lecture 解析 | SFT+DPO+GRPO |
| | MathVerse | EN | 3,940 | ✅ | 数学 VLM 推理 | SFT |
| | MathVista | EN | 1,000 | ✅ | 数学视觉推理 | SFT |
| | TabMWP | EN | 38,000 | ✅ | 表格数学应用题 | SFT |
| | Geometry3K | EN | 3,000 | ✅ | 几何图文，含详细解析 | SFT |
| | CLEVR-Math | EN | 100K | ✅ | 合成图数学推理，增强空间推理 | SFT |
| **图表理解** | ChartQA | EN | 28,000 | ✅ | 柱状图/折线图/饼图问答 | SFT |
| | DVQA | EN | 300K | ✅ | 信息图阅读理解 | SFT |
| **中文理科** | C-Eval | CN | 14,000 | ❌ | 中文理科14个学科 | SFT |
| | CMMLU | CN | 11,917 | ❌ | 中文67学科综合 | SFT |
| | GAOKAO-Bench | CN | 6,000 | ❌ | 高考真题 | SFT |
| | GeoQA+ | CN | 73,000 | ✅ | 中文几何选择题 | SFT |
| **中文数学** | Math23K | CN | 23,000 | ❌ | 中文小学数学应用题 | SFT |
| | Ape210K | CN | 210,000 | ❌ | 中文小学数学大规模 | SFT |
| **科学常识** | AI2D | EN | 5,000 | ✅ | 科学示意图，亲子教育 | SFT |
| | BioVQA | EN | 5,000 | ✅ | 生物图文题 | SFT |
| | TQA | EN | 26,000 | ✅ | 教科书级别图文问答 | SFT |
| **图文通识** | MMMU | EN | 11,000 | ✅ | 30学科大学级图文题 | SFT |
| | OCR-VQA | EN | 200,000 | ✅ | OCR 图文题（降权20-30%） | SFT⚠️ |
| **场景鲁棒** | VizWiz | EN | 32,000 | ✅ | 真实模糊/倾斜场景，增强鲁棒性 | SFT |
| **语言理解** | RACE | CN/EN | 28,000 | ❌ | 中英文阅读理解 | SFT |

## 📋 评估指标矩阵

| 阶段 | 评估命令 | 样本量 | 关键指标 | 绿灯标准 |
|------|----------|--------|----------|----------|
| 训练前 | `--stage baseline` | 500+200 | 基座准确率 / 步骤率 | 保存到 `eval_results/baseline.json` |
| SFT 后 | `--stage sft` | 500+200+200 | 准确率 / 步骤率 / 启发式引导率 | 准确率>50%，步骤率>60% |
| DPO 后 | `--stage dpo` | 50+100+50 | 偏好gap / 准确率不掉 / 退化检测 | gap>0.05，退化<5% |
| GRPO 后 | `--stage grpo` | 50+100+50 | avg_reward / 准确率不掉 / 退化检测 | avg_reward>0.5，退化<5% |
| 细粒度 | `--stage fine` | 100+50 | 五维度雷达评分 | 加权总分>0.5 |
| 最终发布 | `--stage full` | 4241+500 | 准确率 / 步骤率 | 全量 holdout 最终值 |

## 🗂️ 项目架构

```
qwensearch/
├── model/                         # 模型层
│   ├── __init__.py
│   └── qwen_vlm.py               # Qwen2-VL 封装（可插拔基座 + LoRA）
├── dataset/                       # 数据层
│   ├── __init__.py
│   ├── edu_dataset.py            # EduDataset / EduDPODataset / EduGRPODataset
│   └── *.parquet                 # 6个已下载数据集（45,729条）
├── trainer/                       # 训练层
│   ├── __init__.py
│   ├── train_sft.py              # SFT 微调训练
│   ├── train_dpo.py              # DPO 偏好对齐
│   ├── train_grpo.py             # GRPO 强化优化
│   ├── reward_model.py           # EduRewardModel（五维度教育奖励）
│   └── trainer_utils.py          # 训练工具函数
├── scripts/                       # 工具脚本
│   ├── convert_edu_data.py       # 数据集格式转换（支持23种格式）
│   └── web_demo.py               # Gradio Web 演示
├── eval_edu.py                    # 六阶段评估脚本
├── download_data.py               # 一键下载数据集
├── DATA.md                        # 数据集全览
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 1. 创建虚拟环境

<details>
<summary><b>方案A：Conda (推荐)</b></summary>

```bash
# 创建 Python 3.10 虚拟环境
conda create -n qwensearch python=3.10 -y

# 激活环境
conda activate qwensearch

# 安装 PyTorch (CUDA 12.1)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
cd E:\OneDrive\Code\qwensearch
pip install -r requirements.txt
```
</details>

<details>
<summary><b>方案B：venv (轻量)</b></summary>

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 升级 pip
python -m pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```
</details>

> **注意**：Qwen2-VL 需要 `transformers>=4.45.0`，推荐使用 Python 3.10。  
> 如果用 CPU 训练，把 `torch` 改为 `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 2. 下载基座模型

```bash
# 从 ModelScope 下载（国内更快）
pip install modelscope
modelscope download --model qwen/Qwen2-VL-2B-Instruct --local_dir ./model/Qwen2-VL-2B-Instruct

# 或从 HuggingFace 下载
# git lfs install
# git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct ./model/Qwen2-VL-2B-Instruct
```

### 3. 下载数据集

```bash
python download_data.py
```

### 4. 打基线 + 训练 + 评估

```bash
# ① 训练前打基线
python eval_edu.py --stage baseline --max_samples 500

# ② SFT 微调
python trainer/train_sft.py \
    --data_paths "dataset/edu_science.parquet,dataset/edu_math_verse.parquet,dataset/edu_math_vista.parquet,dataset/edu_ocr.parquet,dataset/edu_ceval.parquet,dataset/edu_cmmlu.parquet" \
    --epochs 3 --save_weight edu_sft

# ③ SFT 后评估
python eval_edu.py --stage sft --model_path out/edu_sft --max_samples 500

# ④ DPO 对齐
python trainer/train_dpo.py --data_path dataset/edu_science.parquet --from_weight ../out/edu_sft --epochs 1

# ⑤ DPO 后评估
python eval_edu.py --stage dpo --model_path out/edu_dpo --max_samples 100

# ⑥ GRPO 优化
python trainer/train_grpo.py --data_path dataset/edu_science.parquet --from_weight ../out/edu_dpo --epochs 1

# ⑦ GRPO 后评估
python eval_edu.py --stage grpo --model_path out/edu_grpo --max_samples 100

# ⑧ 最终全量评估
python eval_edu.py --stage full --model_path out/edu_grpo --max_samples -1
```

### 5. Web Demo

```bash
python scripts/web_demo.py --model_path out/edu_grpo
```

## 📜 License

本项目基于 [Apache 2.0](LICENSE) 协议开源。基座模型 Qwen2-VL 遵循其原始协议。

## 🙏 致谢

- [MiniMind-V](https://github.com/jingyaogong/minimind-v) — 项目架构参考
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) — 基座多模态模型
- [ScienceQA](https://scienceqa.github.io/) / [MathVerse](https://mathverse-cuhk.github.io/) / [MathVista](https://mathvista.github.io/) / [C-Eval](https://cevalbenchmark.com/) — 数据集来源