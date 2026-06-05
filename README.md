# QwenSearch — 亲子教育 VLM

<div align="center">

**📸 拍题即答 · 分步引导 · 亲子共学**

基于 Qwen2-VL 基座魔改的拍照做题 VLM，专为亲子教育场景设计。

</div>

---

## ✨ 优化成果（2026-06）

| 优化项 | 改进 | 效果 |
|--------|------|------|
| **训练流程简化** | 移除 DPO 阶段 | SFT → GRPO 两阶段，避免错误偏好信号 |
| **新增中文多学科数据集** | CMMU/CMMMU/M3Exam/MMSciBench | 新增 4 个中文图文做题数据集 |
| **SFT 加权采样** | 平方根倒数加权 | 小数据集采样率提升 12 倍 |
| **GRPO 显存优化** | del tensor + empty_cache() | 显存占用减少 30-40% |
| **GRPO 批处理** | 批量 tokenize | 训练速度提升 20-30% |
| **奖励模型改进** | TF-IDF 语义相似度 | 评分更准确，支持数值近似匹配 |
| **评估指标重设计** | Bootstrap 置信区间 + p-value | 避免统计噪声误判 |
| **重采样公式升级** | v2 平滑公式 + 样本量修正 | 弱项权重不过度，避免过拟合 |
| **新增 8 个工具脚本** | analyze_errors/compare_evals v2/wandb/report | 评估与优化闭环完整 |
| **元评估机制** | 指标一致性 + LLM Judge 框架 | 监控指标本身可靠性 |

**当前数据规模：~222,198 条训练数据（22 个数据集）+ 8,950 条评估数据（19 个数据集）+ 中文图文占比 52.2%**

---

## 🏗️ 训练管线总览（SFT → GRPO 两阶段）

```
┌──────────────┐                            ┌──────────────┐
│   SFT 阶段    │ ──────────────→            │   GRPO 阶段   │
│   监督微调    │   从 SFT 权重初始化策略     │   强化优化    │
├──────────────┤                            ├──────────────┤
│ 使用数据:     │                            │ 使用数据:     │
│ 22个数据集    │                            │ 5K 精选       │
│ ~222K 条      │                            │ CMMU/MMSci等 │
├──────────────┤                            ├──────────────┤
│ Dataset:      │                            │ Dataset:      │
│ EduDataset    │                            │ EduGRPODataset│
├──────────────┤                            ├──────────────┤
│ Loss:         │                            │ Loss:         │
│ CrossEntropy  │                            │ PPO-clip      │
│               │                            │               │
├──────────────┤                            ├──────────────┤
│ 评估:         │                            │ 评估:         │
│ --stage sft   │                            │ --stage grpo  │
│ 19 个评估集   │                            │ 奖励质量(5K)  │
├──────────────┤                            ├──────────────┤
│ 目标指标:     │                            │ 目标指标:     │
│ 准确率>50%    │                            │ avg_reward    │
│ 步骤率>60%    │                            │   >0.5       │
└──────────────┘                            └──────────────┘
         ↓                                           ↓
    保存: edu_sft                              保存: edu_grpo
                                                     ↓
    ┌─────────────────────────────────────────────┐
    │          最终发布评估: --stage full          │
    │  19 个评估集全量 + 引导性分析                 │
    └─────────────────────────────────────────────┘
```

> **💡 为什么去掉了 DPO 阶段？**
> - 当前 DPO 偏好数据采用「简单截断原回答」构造，本质是「长度偏好」而非「质量偏好」
> - GRPO 的奖励模型已能直接优化「引导性」维度，比 DPO 间接学习更直接
> - 避免错误偏好信号风险，训练流程更简洁
> - 详细分析见 [EVAL_DESIGN.md](EVAL_DESIGN.md) 第八章

## 📊 数据集总览（22 个，~222,198 条训练数据 + 8,950 条评估数据）

| 类别 | 数据集 | 语言 | 训练条数 | 评估条数 | 特点 |
|------|--------|:----:|:--------:|:--------:|------|
| **核心中文图文（新增）** | We-Math 2.0 | CN | ~10,000 | 500 | 系统性知识体系 |
| | Geo170K | CN | 50,000 | 500 | 几何推理专项 |
| **中文多学科图文做题（新增）** | CMMU | CN | 1,800 | 200 | K12 7 门核心学科 |
| | CMMMU | CN | 3,000 | 200 | 大学 6 大类 30+ 学科 |
| | M3Exam | CN | 3,000 | 200 | 多语言 K12 三阶段 |
| | MMSciBench | CN | 1,000 | 100 | 中学数学+物理 |
| **核心图文数学** | ScienceQA | EN | 6,218 | 932 | 全理科图文 |
| | MathVerse | EN | 3,940 | 591 | 数学VLM推理 |
| | MathVista | EN | 1,000 | 150 | 数学视觉推理 |
| **中文图文** | windata-math | CN | 10,000 | 500 | 中文数学图文 |
| **OCR 识别** | OCR-VQA | EN | 20,000 | 1,000 | OCR图文题 |
| **图表理解** | ChartQA | EN | 10,000 | 1,000 | 柱状图/折线图/饼图 |
| **中文理科** | C-Eval | CN | 2,654 | 500 | 中文理科14学科 |
| | CMMLU | CN | 11,917 | 1,000 | 中文67学科综合 |
| **中文数学** | Ape210K | CN | 20,000 | 1,000 | 中文小学数学 |
| | OpenR1-Math K12 | CN | 20,000 | 1,000 | 中文K12数学CoT |
| | Gaokao MathQA | CN | 351 | 351 | 高考数学选择题 |
| | Gaokao MathCloze | CN | 118 | 118 | 高考数学填空题 |
| **语言理解** | RACE | CN/EN | 10,000 | 1,000 | 中英文阅读理解 |
| | **合计** | | **~222,198** | **8,950** | |

> **已删除数据集**（GitHub 仓库缺 JSON 元数据或下载不稳定）：CMM-Math、MathReal、We-Math 2.0（备份在 `download_all_data.py` 中已注释）

### 数据分布

| 指标 | 数值 |
|------|------|
| **总训练数据** | ~222,198 条 |
| **总评估数据** | 8,950 条 |
| **中文图文数据** | ~116,000 条 |
| **中文图文占比** | 52.2% |
| **训练数据集数** | 22 个 |
| **评估数据集数** | 19 个 |

## 📦 数据集下载

详细的数据集来源、许可协议、下载方式和使用说明，请参阅 [DATA.md](DATA.md)。

### 一键下载

```bash
# 下载所有训练数据集和评估数据集
python scripts/download_all_data.py

# 仅下载训练数据集
python scripts/download_all_data.py --train

# 仅创建评估数据集（需要先有训练集）
python scripts/download_all_data.py --eval

# 下载指定数据集
python scripts/download_all_data.py --datasets scienceqa ceval
```

### 数据集结构

```
dataset/
├── edu_*.parquet              # 训练数据集 (~222,198 条)
│   ├── edu_science.parquet     # ScienceQA
│   ├── edu_ceval.parquet       # C-Eval
│   ├── edu_cmmu.parquet        # CMMU（中文多学科）
│   ├── edu_geo170k.parquet     # Geo170K（几何推理）
│   └── ... (共 22 个)
│
├── eval/                      # 评估数据集 (8,950 条)
│   ├── eval_science.parquet    # ScienceQA 评估集
│   ├── eval_ceval.parquet      # C-Eval 评估集
│   └── ... (共 19 个)
│
└── ...
```

> **⚠️ 仓库不包含 parquet 文件**（已通过 .gitignore 排除，避免仓库过大）。运行 `python scripts/download_all_data.py` 一键下载。

## 📋 评估方法

### 每次训练后应该怎么评估

| 训练阶段 | 评估方式 | 命令 | 说明 |
|----------|----------|------|------|
| **基线 (baseline)** | 保存基座性能 | `python eval_edu.py --stage baseline --eval_all --max_samples 200` | 记录 19 个评估集基线分数到 `eval_results/baseline.json` |
| **SFT 训练后** | 全量评估（19 个数据集） | `python eval_edu.py --stage sft --model_path out/edu_sft --eval_all --max_samples 200` | 在所有 19 个 holdout 数据集上评估，观察每个数据集的能力变化 |
| **GRPO 训练后** | 四维评估（基础+奖励+细粒度+退化） | `python eval_edu.py --stage grpo --model_path out/edu_grpo --max_samples 200` | 评估 GRPO 奖励质量 + EduRewardModel 五维度细粒度 + 退化检测 |
| **最终发布 (full)** | 全量 holdout | `python eval_edu.py --stage full --model_path out/edu_grpo --eval_all --max_samples 500` | 19 个评估集全量 + ScienceQA test split 全量(4241) |

**👉 核心原则：训练完每个阶段后，立即运行 `--eval_all` 全量评估，对比前后分数变化，确保新阶段没有破坏已有能力。**

### 评估反馈优化（🆕）

训练后的评估结果会自动持久化，支持对比分析和数据重采样：

```bash
# ① 评估自动保存到 eval_results/<stage>_<timestamp>.json
python eval_edu.py --model_path out/edu_sft --stage sft --eval_all --max_samples 200

# ② 对比最新两次评估，查看能力变化
python compare_evals.py

# ③ 基于评估结果生成数据重采样建议（弱项多练）
python scripts/resample_data.py
```

| 工具 | 脚本 | 功能 |
|------|------|------|
| **结果对比** | `compare_evals.py` | 对比两次评估的指标变化（✅上升/⚠️下降），自动定位退化 |
| **数据重采样** | `scripts/resample_data.py` | 分析各数据集得分，得分低的数据集自动提高训练权重 |
| **结果持久化** | `eval_edu.py`（已集成）| 所有阶段评估结果自动保存到 `eval_results/`，附带时间戳 |

### 19 个评估数据集

| 评估数据集 | 评估命令 | 条数 | 评估指标 |
|-----------|----------|------|----------|
| **中文核心图文数学（新增）** | | | |
| We-Math 2.0 | `--eval_dataset we_math` | 500 | 选项匹配率 + 步骤完整率 |
| Geo170K | `--eval_dataset geo170k` | 500 | 答案匹配率 + 步骤完整率 |
| windata-math | `--eval_dataset windata_math` | 500 | 答案匹配率 |
| **中文多学科图文做题（新增）** | | | |
| CMMU | `--eval_dataset cmmu` | 200 | 选项匹配率 + 步骤完整率 |
| CMMMU | `--eval_dataset cmmmu` | 200 | 选项匹配率 |
| M3Exam | `--eval_dataset m3exam` | 200 | 选项匹配率 |
| MMSciBench | `--eval_dataset mmscibench` | 100 | 答案匹配率 + 步骤完整率 |
| **核心图文数学** | | | |
| ScienceQA | `--eval_dataset scienceqa` | 932 | 答案准确率 + 步骤完整率 + 启发式引导率 |
| MathVerse | `--eval_dataset math_verse` | 591 | 关键词匹配率 |
| MathVista | `--eval_dataset math_vista` | 150 | 答案匹配率 |
| **OCR 识别与图表** | | | |
| OCR-VQA | `--eval_dataset ocr` | 1,000 | 关键词匹配率 |
| ChartQA | `--eval_dataset chartqa` | 1,000 | 答案匹配率 |
| **中文理科与数学** | | | |
| C-Eval | `--eval_dataset ceval` | 500 | 选项匹配率 |
| CMMLU | `--eval_dataset cmmlu` | 1,000 | 选项匹配率 |
| Ape210K | `--eval_dataset ape210k` | 1,000 | 关键词匹配率 |
| OpenR1-Math | `--eval_dataset openr1_math` | 1,000 | 关键词匹配率 + 步骤完整率 |
| Gaokao MathQA | `--eval_dataset gaokao_mathqa` | 351 | 选项匹配率 |
| Gaokao MathCloze | `--eval_dataset gaokao_mathcloze` | 118 | 数值匹配率 |
| **语言理解** | | | |
| RACE | `--eval_dataset race` | 1,000 | 选项匹配率 |
| **合计** | | **~8,950** | |

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
│   ├── edu_dataset.py            # 数据集加载器
│   └── eval/                     # 评估数据集
├── scripts/                      # 工具脚本
│   ├── __init__.py
│   ├── download_all_data.py       # 一键下载所有数据集
│   ├── convert_edu_data.py        # 数据集格式转换
│   ├── convert_windata.py         # windata 数据转换
│   ├── create_eval_sets.py        # 创建评估集
│   ├── web_demo.py                # Web 演示
│   │
│   ├── eval/                      # 🆕 评估相关脚本
│   │   ├── __init__.py
│   │   ├── edu_evaluate.py        # 一站式评估入口（run/compare/errors/meta/report/all）
│   │   ├── eval_edu.py            # 主评估脚本（含 Bootstrap 置信区间）
│   │   ├── compare_evals.py       # 评估结果对比 v2（含 95%CI + p-value）
│   │   ├── analyze_errors.py      # 错误案例分析（自动归类）
│   │   ├── meta_evaluation.py     # 元评估（指标一致性 + LLM Judge）
│   │   └── generate_report.py     # 自动生成 Markdown 评估报告
│   │
│   └── optimize/                  # 🆕 优化相关脚本
│       ├── __init__.py
│       ├── edu_optimize.py        # 一站式优化入口（resample/build/retrain/auto）
│       ├── resample_data.py       # 数据重采样 v2（平滑公式 + 样本量修正）
│       ├── build_preference_data.py  # GRPO 强化数据准备（5K 精选）
│       └── wandb_integration.py   # 训练监控（wandb/SwanLab/TensorBoard）
├── trainer/                      # 训练层
│   ├── __init__.py
│   ├── train_sft.py              # SFT 微调训练（22 个数据集，加权采样）
│   ├── train_grpo.py             # GRPO 强化优化（从 SFT 衔接，5K 精选）
│   ├── reward_model.py           # 教育奖励模型（五维度细粒度）
│   └── trainer_utils.py          # 训练工具
├── EVAL_DESIGN.md               # 评估系统设计文档
├── DATA.md                      # 数据集说明
├── README.md
└── requirements.txt
```

## 📋 项目要求

### 🖥️ 硬件要求

| 配置项 | 最低要求 | 推荐配置 | 最佳配置 |
|--------|---------|---------|---------|
| **GPU** | NVIDIA RTX 3090 (24GB) | NVIDIA A100 (40GB) | NVIDIA A100 (80GB) / H100 |
| **显存** | 24 GB | 40 GB | 80 GB |
| **内存** | 32 GB | 64 GB | 128 GB |
| **硬盘** | 100 GB SSD | 200 GB NVMe | 500 GB NVMe |
| **CUDA** | 11.8 | 12.1 | 12.1+ |
| **网络** | 50 Mbps | 100 Mbps | 500 Mbps+（含学术代理） |

> **⚠️ 关于显存**：
> - **2B 模型 + LoRA**：24 GB 显存可跑（推荐）
> - **2B 模型全量微调**：需 40 GB+ 显存
> - **7B 模型**：至少 40 GB 显存（LoRA）

### 📦 软件依赖

| 依赖 | 版本要求 | 必需/可选 | 说明 |
|------|---------|:---------:|------|
| **Python** | 3.10+ | 必需 | 推荐 3.10 |
| **PyTorch** | 2.4.0+ | 必需 | CUDA 12.1 对应 cu121 |
| **Transformers** | ≥4.45.0 | 必需 | Qwen2-VL 要求 |
| **Datasets** | ≥2.20.0 | 必需 | 数据加载 |
| **PEFT** | ≥0.10.0 | 必需 | LoRA 训练 |
| **accelerate** | ≥0.30.0 | 必需 | 分布式训练 |
| **Pillow** | ≥10.0.0 | 必需 | 图像处理 |
| **pyarrow** | ≥15.0.0 | 必需 | parquet 读写 |
| **sentencepiece** | ≥0.2.0 | 必需 | tokenizer |
| **einops** | 最新 | 必需 | 张量操作 |
| **tiktoken** | 最新 | 必需 | tokenizer |
| **vLLM** | ≥0.5.0 | 可选 | 推理加速 |
| **wandb** | 最新 | 可选 | 训练监控 |
| **swanlab** | 最新 | 可选 | 国产训练监控 |
| **tensorboard** | 最新 | 可选 | 训练可视化 |

**完整安装**：
```bash
# 基础依赖
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.45.0 datasets>=2.20.0 peft>=0.10.0 accelerate>=0.30.0
pip install Pillow>=10.0.0 pyarrow>=15.0.0 sentencepiece>=0.2.0 einops tiktoken

# 可选依赖
pip install vllm>=0.5.0 wandb swanlab tensorboard
```

### ⏱️ 预估时间

| 阶段 | 数据规模 | RTX 3090 (24GB) | A100 (40GB) | A100 (80GB) |
|------|---------|:---:|:---:|:---:|
| **数据下载** | 22 个数据集 | ~3-5 小时 | ~3-5 小时 | ~3-5 小时 |
| **SFT 训练** | ~222K 样本 (3 epochs) | ~36-48 小时 | ~12-18 小时 | ~8-12 小时 |
| **GRPO 训练** | 5K 精选 (1 epoch) | ~6-10 小时 | ~2-4 小时 | ~1-2 小时 |
| **全量评估** | 19 个数据集 × 200 | ~3-5 小时 | ~1-2 小时 | ~1 小时 |
| **报告生成** | - | < 5 分钟 | < 5 分钟 | < 5 分钟 |
| **总耗时** | 完整流程 | **~50-70 小时** | **~18-28 小时** | **~12-18 小时** |

> **📊 测试配置**：
> - `batch_size=4, gradient_accumulation=8`（有效 batch=32）
> - `max_seq_len=2048, learning_rate=1e-4`
> - 评估 `--max_samples=200/集合`（默认）
>
> **⚠️ 注意**：
> - 数据下载受网络速度影响很大，国内推荐用 ModelScope 镜像
> - 全量评估 `--eval_all --max_samples -1` 时长会增加 5-10 倍
> - GRPO 训练强烈推荐 40GB+ 显存

### 🔒 评估数据集与训练集严格分离

**这是机器学习评估的基本原则**：评估集的数据**绝对不能出现在训练集**中。

#### ✅ 实现方式（`download_all_data.py` 中）

1. **训练集先从原始数据集采样**
2. 然后从训练集中**真正分离**评估样本（不是简单采样）
3. 写评估集到 `dataset/eval/eval_*.parquet`
4. **写回训练集**到 `dataset/edu_*.parquet`（已移除评估样本）
5. 每个样本添加 `split='train'/'eval'` 标记

#### 🔍 验证方式

```bash
# 检查训练集和评估集是否重叠（应输出 0）
python -c "
import pyarrow.parquet as pq
train = pq.read_table('dataset/edu_science.parquet').to_pylist()
eval_set = pq.read_table('dataset/eval/eval_science.parquet').to_pylist()
train_ids = set(r.get('id', r.get('question', '')[:100]) for r in train)
eval_ids = set(r.get('id', r.get('question', '')[:100]) for r in eval_set)
overlap = train_ids & eval_ids
print(f'训练样本数: {len(train_ids)}')
print(f'评估样本数: {len(eval_ids)}')
print(f'重叠样本数: {len(overlap)}  (应为 0)')
"
```

#### ⚠️ 重要：升级提示

如果你是**从旧版本升级**，训练集中可能仍含评估样本。

**解决方法**（二选一）：
```bash
# 方法 1：删除后重新下载（推荐）
# Windows PowerShell:
Remove-Item dataset\edu_*.parquet, dataset\eval\eval_*.parquet
python download_all_data.py

# 方法 2：使用辅助脚本
python scripts/eval/check_split_integrity.py  # 检测
python download_all_data.py --force_resplit  # 强制重分离
```

> 💡 **设计原因**：
> - 一些数据集（如 OCR-VQA、Ape210K）没有官方的 test split，必须自己划分
> - 一些数据集（ScienceQA、MathVista）有官方 test split，**优先使用官方 split**（在 `convert_edu_data.py` 中已实现）
> - 对于无官方 split 的数据集，`create_eval_set()` 函数会做"真分离"（写回训练集时移除评估样本）

---

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
# 一键下载所有 22 个训练数据集和 19 个评估数据集
python scripts/download_all_data.py

# 或分步下载
python scripts/download_all_data.py --train    # 仅训练集
python scripts/download_all_data.py --eval     # 仅评估集
```

### 4. 一站式训练 + 评估 + 优化（SFT → GRPO 两阶段）

> **💡 推荐**：用一站式脚本完成整个流程
> 完整流程只需 **5 个命令**（SFT → 评估 → 优化 → GRPO → 最终评估）

```bash
# ① SFT 训练（22 个数据集，加权采样）
python trainer/train_sft.py --epochs 3 --save_weight edu_sft

# ② 一站式 SFT 评估（run + meta + report 三合一，自动生成 report.md）
python scripts/eval/edu_evaluate.py all --stage sft --model_path out/edu_sft --eval_all

# ③ 一站式优化（resample + retrain 两合一，自动使用 v2 平滑公式）
python scripts/optimize/edu_optimize.py auto --epochs 2 --save_weight edu_sft_v2

# ④ GRPO 训练（直接从 SFT 衔接，跳过 DPO）
python trainer/train_grpo.py --from_weight ../out/edu_sft --epochs 1

# ⑤ 一站式最终评估（run + meta + report + 全量 19 个数据集）
python scripts/eval/edu_evaluate.py all --stage full --model_path out/edu_grpo --eval_all
```

> **🎯 关键说明**：
> - `edu_evaluate.py all` = 评估 + 元评估 + 生成报告（三合一）
> - `edu_optimize.py auto` = 重采样 + 再训练（两合一）
> - 整个训练 + 评估 + 优化流程只需 **5 个命令**即可完成

### 5. 单独工具（按需使用）

如果想精细控制每一步，可以用单独的命令：

```bash
# === 评估子命令（scripts/eval/edu_evaluate.py）===
python scripts/eval/edu_evaluate.py run     --stage baseline --eval_all
python scripts/eval/edu_evaluate.py run     --stage sft --model_path out/edu_sft --eval_all
python scripts/eval/edu_evaluate.py compare --show_weak
python scripts/eval/edu_evaluate.py errors  --output_errors errors.json
python scripts/eval/edu_evaluate.py meta    --check_consistency
python scripts/eval/edu_evaluate.py report  --output report.md

# === 优化子命令（scripts/optimize/edu_optimize.py）===
python scripts/optimize/edu_optimize.py resample --output weights.json
python scripts/optimize/edu_optimize.py build    --output edu_grpo.parquet
python scripts/optimize/edu_optimize.py retrain  --data_paths "..." --epochs 2
```

### 6. 训练监控（🆕）

支持 wandb / SwanLab / TensorBoard / 本地 JSON 自动降级：

```python
# 在训练脚本中集成
from scripts.optimize.wandb_integration import TrainingLogger

logger = TrainingLogger(backend="auto", project="qwensearch", config={...})
for step, batch in enumerate(dataloader):
    loss = train_step(model, batch)
    logger.log({"train/loss": loss, "train/lr": scheduler.get_last_lr()[0]}, step=step)
logger.log_artifact("out/edu_sft/pytorch_model.bin", name="model")
logger.finish()
```

### 7. Web Demo

```bash
python scripts/web_demo.py --model_path out/edu_grpo
```

## � 相关文档

- [DATA.md](DATA.md) — 数据集详细说明（22 个数据集的来源/许可/特点/下载）
- [EVAL_DESIGN.md](EVAL_DESIGN.md) — 评估系统设计文档（19 个评估集 + 4 阶段评估 + 8 个工具脚本）

## �� License

本项目基于 [Apache 2.0](LICENSE) 协议开源。基座模型 Qwen2-VL 遵循其原始协议。

## 🙏 致谢

- [MiniMind-V](https://github.com/jingyaogong/minimind-v) — 项目架构参考，README 风格借鉴
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) — 基座多模态模型
- [ScienceQA](https://scienceqa.github.io/) / [MathVerse](https://mathverse-cuhk.github.io/) / [MathVista](https://mathvista.github.io/) / [C-Eval](https://cevalbenchmark.com/) / [CMMU](https://github.com/FlagOpen/CMMU) / [CMMMU](https://github.com/m-a-p/CMMMU) / [M3Exam](https://github.com/DAMO-NLP-SG/M3Exam) / [MMSciBench](https://github.com/XinwuYe/MMSciBench) — 数据集来源
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) — 数据加载框架