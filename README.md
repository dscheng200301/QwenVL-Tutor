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

## 📊 数据集总览（12 个，106,198 条）

| 类别 | 数据集 | 语言 | 条数 | 含图 | 用途 | 阶段 |
|------|--------|:----:|:----:|:----:|------|:----:|
| **核心图文数学** | ScienceQA | EN | 6,218 | ✅ | 全理科图文，带 lecture 解析 | SFT+DPO+GRPO |
| | MathVerse | EN | 3,940 | ✅ | 数学 VLM 推理 | SFT |
| | MathVista | EN | 1,000 | ✅ | 数学视觉推理 | SFT |
| **OCR 识别** | OCR-VQA | EN | 20,000 | ✅ | OCR 图文题（降权20-30%） | SFT⚠️ |
| **图表理解** | ChartQA | EN | 10,000 | ✅ | 柱状图/折线图/饼图问答 | SFT |
| **中文理科** | C-Eval | CN | 2,654 | ❌ | 中文理科14个学科 | SFT |
| | CMMLU | CN | 11,917 | ❌ | 中文67学科综合 | SFT |
| **中文数学** | Ape210K | CN | 20,000 | ❌ | 中文小学数学大规模 | SFT |
| | **OpenR1-Math K12** 🆕 | CN | 20,000 | ❌ | 中文 K12 数学推理链（CoT）| SFT |
| | **Gaokao MathQA** 🆕 | CN | 351 | ❌ | 高考数学选择题 | SFT |
| | **Gaokao MathCloze** 🆕 | CN | 118 | ❌ | 高考数学填空题 | SFT |
| **语言理解** | RACE | CN/EN | 10,000 | ❌ | 中英文阅读理解 | SFT |
| | **合计** | | **106,198** | | | |

## 📋 评估方法

### 每次训练后应该怎么评估

| 训练阶段 | 评估方式 | 命令 | 说明 |
|----------|----------|------|------|
| **SFT 训练后** | 全量评估（12 个数据集） | `python eval_edu.py --model_path out/edu_sft --eval_all --max_samples 500` | 在所有 holdout 数据集上评估，观察每个数据集的能力变化 |
| **DPO 训练后** | 重点检查 3 个核心集 | `--eval_dataset scienceqa` + `--eval_dataset openr1_math` + `--eval_dataset cmmlu` | DPO 只增强偏好对齐，检查基础能力是否退化 |
| **GRPO 训练后** | 细粒度分析 | `--eval_dataset scienceqa --max_samples 200` | 检查五维度奖励（准确性/完整性/流畅性/引导性/规范性） |

**👉 核心原则：训练完每个阶段后，立即运行 `--eval_all` 全量评估，对比前后分数变化，确保新阶段没有破坏已有能力。**

### 12 个评估数据集

| 评估数据集 | 评估命令 | 条数 | 评估指标 |
|-----------|----------|------|----------|
| ScienceQA | `--eval_dataset scienceqa` | 932 | 答案准确率 + 步骤完整率 + 启发式引导率 |
| C-Eval | `--eval_dataset ceval` | 500 | 选项匹配率 |
| OCR-VQA | `--eval_dataset ocr` | 1,000 | 关键词匹配率 |
| Ape210K | `--eval_dataset ape210k` | 1,000 | 关键词匹配率 |
| OpenR1-Math | `--eval_dataset openr1_math` | 1,000 | 关键词匹配率 + 步骤完整率 |
| ChartQA | `--eval_dataset chartqa` | 1,000 | 答案匹配率 |
| CMMLU | `--eval_dataset cmmlu` | 1,000 | 选项匹配率 |
| MathVerse | `--eval_dataset math_verse` | 591 | 关键词匹配率 |
| MathVista | `--eval_dataset math_vista` | 150 | 答案匹配率 |
| RACE | `--eval_dataset race` | 1,000 | 选项匹配率 |
| Gaokao MathQA | `--eval_dataset gaokao_mathqa` | 351 | 选项匹配率 |
| Gaokao MathCloze | `--eval_dataset gaokao_mathcloze` | 118 | 数值匹配率 |

### 评估指标说明

| 指标 | 计算方法 | 适用数据集 |
|------|----------|-----------|
| 答案准确率 | 检查模型回复中是否包含正确答案 | ScienceQA, MathVista, Gaokao |
| 选项匹配率 | 检查答案选项字母是否出现 | C-Eval, CMMLU, RACE |
| 关键词匹配率 | GT 答案中关键词与回复的交集比率 | OCR-VQA, Ape210K, MathVerse, OpenR1-Math |
| 步骤完整率 | 回复中是否包含分步推理关键词 | 所有图文数学数据集 |
| 启发式引导率 | 回复中是否包含引导性语言 | 亲子教育场景评估 |

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