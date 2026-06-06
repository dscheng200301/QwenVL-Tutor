# QwenSearch — 亲子教育 VLM

<div align="center">

**📸 拍题即答 · 分步引导 · 亲子共学**

基于 Qwen2-VL 基座魔改的拍照做题 VLM，专为亲子教育场景设计

</div>

---

## ✨ 核心特性

- **🎯 22 个训练数据集**：~222K 条样本，中文图文占比 **52.2%**
- **🚀 两阶段训练**：SFT → GRPO（移除 DPO，避免错误偏好信号）
- **📊 19 个评估数据集**：Bootstrap 置信区间 + p-value 显著性检验
- **🔧 一站式工具**：`edu_evaluate.py all` + `edu_optimize.py auto`
- **🔒 严格数据分离**：评估集数据**绝不进入**训练集

---

## 📋 项目要求

### 🖥️ 硬件

| 配置 | 最低 | 推荐 | 最佳 |
|------|------|------|------|
| GPU | RTX 3090 (24GB) | A100 (40GB) | A100 (80GB) / H100 |
| 显存 | 24 GB | 40 GB | 80 GB |
| 内存 | 32 GB | 64 GB | 128 GB |
| 硬盘 | 100 GB SSD | 200 GB NVMe | 500 GB NVMe |

### 📦 核心依赖

```bash
pip install -r requirements.txt
```

**完整依赖**见 [`requirements.txt`](requirements.txt)，含：

| 类别 | 包 |
|------|---|
| 核心 ML | torch, transformers, accelerate, peft |
| 数据 | datasets, pyarrow, Pillow, numpy |
| 分布式训练（可选） | deepspeed |
| 推理加速（可选） | vllm |
| 监控 | wandb, swanlab |
| 演示 | gradio |

> vLLM 不可用时 `edu_evaluate.py` 会自动降级到 HuggingFace transformers

### ⏱️ 预估时间（完整流程）

| GPU | SFT (3 ep) | GRPO (1 ep) | 评估 (19×200) | 总计 |
|-----|:----------:|:-----------:|:-------------:|:----:|
| RTX 3090 | 36-48 h | 6-10 h | 3-5 h | **50-70 h** |
| A100 40GB | 12-18 h | 2-4 h | 1-2 h | **18-28 h** |
| A100 80GB | 8-12 h | 1-2 h | 1 h | **12-18 h** |

> 配置：`batch_size=4 × grad_accum=8`（有效 batch=32），`max_seq_len=2048`

---

## 🏗️ 训练管线（SFT → GRPO 两阶段）

```
SFT 训练 (22 datasets, ~222K, weighted sampling)
    ↓
一站式评估 (edu_evaluate.py all)  → 自动生成 report.md
    ↓
一站式优化 (edu_optimize.py auto) → resample + retrain
    ↓
GRPO 训练 (5K curated, 5 维度奖励)  ← 从 SFT 衔接
    ↓
一站式最终评估 → 19 datasets + ScienceQA test split
```

**为什么去掉 DPO？** 当前 DPO 偏好数据用「简单截断」构造，本质是长度偏好而非质量偏好，训练风险大于收益。GRPO 的奖励模型已能直接优化「引导性」维度。

---

## 📊 数据集概览

| 类别 | 数量 | 代表数据集 | 训练 / 评估 |
|------|:----:|-----------|:-----------:|
| 中文核心图文数学 | 3 | We-Math 2.0, Geo170K, windata-math | ~70K / 1.5K |
| 中文多学科图文做题 | 4 | CMMU, CMMMU, M3Exam, MMSciBench | ~8.8K / 0.7K |
| 核心图文数学 | 3 | ScienceQA, MathVerse, MathVista | ~11K / 1.7K |
| OCR / 图表 | 2 | OCR-VQA, ChartQA | ~30K / 2K |
| 中文理科 / 数学 | 5 | C-Eval, CMMLU, Ape210K, OpenR1-Math, Gaokao | ~55K / 3K |
| 语言理解 | 1 | RACE | 10K / 1K |
| **合计** | **22** | — | **~222K / 8,950** |

详细说明（来源、许可、下载）：**[DATA.md](DATA.md)**

---

## 🗂️ 项目结构

```
qwensearch/
├── model/                     # Qwen2-VL 封装 + LoRA
├── dataset/                   # 训练/评估数据（运行 download_all_data.py 下载）
├── scripts/
│   ├── eval/                  # 🆕 评估工具
│   │   ├── edu_evaluate.py    #   一站式入口（6 子命令）
│   │   ├── eval_edu.py        #   主评估（19 数据集 + 置信区间）
│   │   ├── compare_evals.py   #   对比 + p-value
│   │   ├── analyze_errors.py  #   错误归类
│   │   ├── meta_evaluation.py #   指标一致性
│   │   └── generate_report.py #   Markdown 报告
│   ├── optimize/              # 🆕 优化工具
│   │   ├── edu_optimize.py    #   一站式入口（4 子命令）
│   │   ├── resample_data.py   #   v2 平滑公式
│   │   └── wandb_integration.py
│   ├── download_all_data.py   # 一键下载（严格分离 eval）
│   ├── convert_edu_data.py    # 22 数据集转换
│   └── web_demo.py
├── trainer/                   # 训练层
│   ├── train_sft.py
│   ├── train_grpo.py
│   └── reward_model.py
├── EVAL_DESIGN.md             # 评估系统设计
├── DATA.md                    # 数据集详细说明
└── requirements.txt
```

---

## 🚀 快速开始

### 1. 环境

```bash
# Conda（推荐）
conda create -n qwensearch python=3.10 -y && conda activate qwensearch
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 或 venv
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 下载基座模型

```bash
pip install modelscope
modelscope download --model qwen/Qwen2-VL-2B-Instruct --local_dir ./model/Qwen2-VL-2B-Instruct
```

### 3. 下载数据集

```bash
python download_all_data.py    # 一键下载 22 训练 + 19 评估
```

### 4. 训练 + 评估 + 优化（一站式 5 步）

> **🆕 自动加速**：以下命令都默认自动检测 GPU 数量，多卡时自动启用 DDP/DeepSpeed，单卡直接训练

```bash
# ① SFT 训练（多卡自动 DDP，单卡直接跑）
python trainer/train_sft.py --epochs 3 --save_weight edu_sft

# ② 一站式 SFT 评估（自动 vLLM 加速）
python scripts/eval/edu_evaluate.py all --stage sft --model_path out/edu_sft --eval_all

# ③ 一站式优化（多卡自动启用 DeepSpeed）
python scripts/optimize/edu_optimize.py auto --epochs 2 --save_weight edu_sft_v2

# ④ GRPO 训练（直接从 SFT 衔接）
python trainer/train_grpo.py --from_weight ../out/edu_sft --epochs 1

# ⑤ 一站式最终评估
python scripts/eval/edu_evaluate.py all --stage full --model_path out/edu_grpo --eval_all
```

### 5. 单独工具（按需）

```bash
# 评估子命令
python scripts/eval/edu_evaluate.py {run|compare|errors|meta|report} [...]

# 优化子命令
python scripts/optimize/edu_optimize.py {resample|build|retrain|auto} [...]

# Web Demo
python scripts/web_demo.py --model_path out/edu_grpo
```

### 6. 评估报告命名（自动带时间 + 效果）

每次评估生成的报告**自动命名**为 `eval_results/{stage}_{YYYYMMDD}_{HHMM}_acc{X.XXXX}.md`：

```
eval_results/
├── sft_20260605_1430_acc0.6123.md       # 2026-06-05 14:30 训练后评估，平均准确率 0.6123
├── grpo_20260605_1530_acc0.6845.md      # GRPO 强化后评估
├── full_20260606_0900_acc0.6912.md      # 最终发布评估
└── compare_20260606_1000_acc0.6500.md   # 对比评估
```

**手动指定**（可选）：
```bash
python scripts/eval/edu_evaluate.py report --output my_report.md
```

> 命名格式：`{阶段}_{年月日}_{时分}_acc{平均准确率}.md`，方便归档和对比

### 7. GRPO 奖励模型（EduRewardModel）

> **位置**：`trainer/reward_model.py`

GRPO 训练用 **`EduRewardModel`**（规则化奖励，无需训练 NN），五维度评分（总分 0~1）：

| 维度 | 权重 | 计算方法 |
|------|:----:|----------|
| 答案准确性 | 0.30 | 关键词匹配 + TF-IDF 语义相似度 + 数值容差 |
| 步骤完整性 | 0.25 | 解题步骤结构词（"第一步"、"然后"、"所以"等） |
| 启发式引导 | 0.20 | 引导词检测 - 直接给答案的负面模式 |
| 语言流畅度 | 0.15 | 中英文表达流畅度 |
| 格式规范性 | 0.10 | "答案是 X"、"最终答案为 X" 等格式 |

**配套函数**：

- `EduRewardModel.compute_reward(response, gt)` → 单条 reward
- `EduRewardModel.compute_group_rewards(responses, gt)` → 组内 K 个候选的 rewards
- `edu_grpo_advantage(rewards)` → 组内标准化优势
- `edu_grpo_policy_loss(...)` → GRPO 策略损失（含 KL 散度）

**核心设计**：

```python
# 引导关键词（正面）
SCAFFOLDING_KEYWORDS = ["观察", "思考", "想一想", "你能发现", ...]

# 直接给答案（负面）
DIRECT_ANSWER_PATTERNS = [r"^[A-D][\.\)]\s*\n", r"^答案[是为]：?[A-D]", ...]

# 数字同义词映射
NUMBER_KEYWORDS = {"一": "1", "二": "2", "两": "2", "加倍": "×2", ...}
```

**特点**：规则化（不训练 NN）、轻量级（TF-IDF 相似度无 GPU 开销）、中文优化、教育领域专用。

> 💡 **质量更高？** 可选用 `LLMRewardModel`（LLM-as-Judge）替代规则模型，支持 API / 本地 vLLM / 混合三种后端。详见 [README §8 LLM-as-Judge 奖励](#8-llm-as-judge-奖励可选)。

### 8. LLM-as-Judge 奖励（可选）

> **位置**：`trainer/llm_reward.py`

如果觉得 `EduRewardModel` 评分质量不够（规则化模型准确率约 70-75%），可使用 LLM-as-Judge 替代（准确率 85-92%）。

#### 4 种后端对比

| 后端 | 成本 | 速度 | 质量 | 硬件 |
|------|------|------|------|------|
| `APILLMRewardModel` | 💰💰 | 中 | ⭐⭐⭐⭐⭐ | 任意 |
| `LocalVLLMRewardModel` | 0（一次性） | ⚡⚡⚡ | ⭐⭐⭐⭐ | 24GB+ 显存 |
| `LocalHFRewardModel` | 0 | 慢 | ⭐⭐⭐ | 任意 |
| `HybridLLMRewardModel` | 0/💰 | 中 | ⭐⭐⭐⭐ | 任意 |

#### 使用方式

```python
from trainer.llm_reward import create_llm_reward

# 自动选择（优先 vLLM > HF > API）
reward = create_llm_reward(backend="auto")

# 方式 1: API（GPT-4o-mini，约 $3-5/全量训练）
reward = create_llm_reward(
    backend="api",
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o-mini",
)

# 方式 2: 本地 Qwen2.5-72B-AWQ（推荐，0 成本）
reward = create_llm_reward(
    backend="vllm",
    model_path="Qwen/Qwen2.5-72B-Instruct-AWQ",
    tensor_parallel_size=1,
)

# 方式 3: 混合（LLM 70% + 规则 20% + 格式 10%，防 reward hacking）
from trainer.reward_model import EduRewardModel
reward = create_llm_reward(
    backend="hybrid",
    llm_model=create_llm_reward(backend="vllm"),
    rule_model=EduRewardModel(),
)

# 替换 train_grpo.py 中的 EduRewardModel
# reward = EduRewardModel()
reward = create_llm_reward(backend="hybrid")
scores = reward.compute_group_rewards(responses, gts, questions)
```

#### 评分维度（与 EduRewardModel 一致）

| 维度 | 权重 | 说明 |
|------|:----:|------|
| 答案准确性 | 0.40 | 与标准答案一致性（LLM 严格比对） |
| 步骤完整性 | 0.20 | 是否给出清晰步骤 |
| 启发式引导 | 0.20 | 引导式提问 vs 直接给答案 |
| 语言流畅度 | 0.10 | 中文表达质量 |
| 格式规范性 | 0.10 | "答案是 X" 格式 |

#### 实施建议

1. **短期**：先用 `EduRewardModel` 跑通流程（小规模验证）
2. **中期**：用 `LocalVLLMRewardModel` 替代（成本 0，质量大幅提升）
3. **长期**：用 `HybridLLMRewardModel`（防 reward hacking，最稳健）

> ⚠️ **GRPO 中 LLM 必须冻结**：训练时只用 LLM 评分，**不能让 LLM 一起更新**，否则 reward 会偏向自己。LLM 在训练中仅作为"评分员"。

> 详见 `trainer/llm_reward.py` 源码（~400 行）。

---

## 🖥️ 终端实时可视化

> 训练 / 评估 / 优化过程自动显示**实时进度窗口**：进度条、loss 曲线、GPU 显存、ETA。
> 无需手动配置，rich 库（已在 `requirements.txt`）默认启用。

### 自动效果

**训练时** 自动显示：

```
┌─ SFT Training ─────────────────────────────────────┐
│ ⠋ Training ████████████████░░░░░░░░ 156/200  • 02:30  • ETA 01:00 │
│                                                     │
│ 📊 实时指标                                         │
│   train/loss          1.2345  ↓ -0.0234            │
│   train/learning_rate  0.0001  → 平稳              │
│   train/throughput     4.23    ↑ +0.10              │
│                                                     │
│ 💻 系统状态                                         │
│   GPU 0 显存    20.5 / 40.0 GB (51%)              │
│   已用时        150 秒                             │
└─────────────────────────────────────────────────────┘
```

**评估时** 自动显示：

```
Evaluation 开始，共 19 个数据集

▶ scienceqa (200 样本)
  [████████████████████░░░░░░] 156/200 (78%) acc=0.6123
▶ math_vista (150 样本)
  [██████████████████████░░░] 142/150 (94%) acc=0.5342
...

✓ Evaluation 完成
┌─ 评估结果汇总 ─────────────┐
│ 数据集     准确率   样本数  │
│ scienceqa  0.6123   200     │
│ math_vista 0.5342   150     │
│ ...                        │
│ 平均       0.5823   —       │
└────────────────────────────┘
```

**优化时** 自动显示权重变化柱状图：

```
Optimization 数据集权重（Top 10）:
┌──────────────────────┬────────┬──────────────────────┐
│ 数据集               │ 权重   │ 条形                 │
│ cmmu                 │ 2.10   │ ████████████████     │
│ scienceqa            │ 1.85   │ ██████████████       │
│ ...                  │        │                      │
└──────────────────────┴────────┴──────────────────────┘
```

### 关闭（如不需要）

环境变量 `QWENSEARCH_NO_DASHBOARD=1` 即可关闭，回到普通 print 输出：

```bash
# 训练时关闭
QWENSEARCH_NO_DASHBOARD=1 python trainer/train_sft.py --epochs 3

# 评估时关闭
QWENSEARCH_NO_DASHBOARD=1 python scripts/eval/edu_evaluate.py all ...
```

> 实现位于 [`trainer/terminal_dashboard.py`](trainer/terminal_dashboard.py)，不依赖 wandb，离线运行也可用。

---

## 📈 wandb 训练监控

> 用 wandb 记录训练 / 评估 / 优化全过程的曲线，便于对比和调参。
> 首次使用需 `wandb login`；国内网络可用 `swanlab` 替代（pip install swanlab）。

### 启用 wandb

```bash
# 训练时启用
python trainer/train_sft.py --use_wandb --wandb_project QwenSearch
python trainer/train_grpo.py --use_wandb --wandb_project QwenSearch

# 通过一站式脚本启用
python scripts/optimize/edu_optimize.py retrain --use_wandb --epochs 2
```

### 各阶段要监控的指标

| 阶段 | 关键指标 | 作用 |
|------|----------|------|
| **SFT 训练** | `train/loss` / `train/learning_rate` / `train/throughput` / `train/memory_gb` | 训练是否收敛、是否 OOM |
| **GRPO 训练** | `train/grpo_loss` / `train/reward_mean` / `train/reward_std` / `train/throughput` | 奖励是否提升、分布是否合理 |
| **评估** | `eval/accuracy_<dataset>` / `eval/avg_reward` | 各数据集表现 + 平均能力 |
| **优化（重采样）** | `optim/weight_<dataset>` | 弱项数据集权重变化 |
| **系统** | GPU 利用率 / 显存 / 吞吐 | 性能调优 |

### 推荐曲线对比

1. **训练 vs 验证损失**：SFT 后期应平稳下降，**无过拟合**（训练损失↓，验证损失↑=过拟合）
2. **GRPO 奖励曲线**：`train/reward_mean` 应稳步上升，`train/reward_std` 不应过大（否则样本间奖励差异大）
3. **各数据集准确率**：评估后 `eval/accuracy_<dataset>` 曲线，可识别弱项

### 命令速查

```bash
# 登录 wandb
wandb login

# 查看训练运行
wandb sweep / WANDB_PROJECT=QwenSearch wandb graph

# 对比多个 run
wandb compare --project QwenSearch
```

> 训练脚本已内置 wandb 集成，启用后无需额外配置。GPU 监控建议用 `nvidia-smi dmon` 或 `wandb` 的 system metrics。

---

## 🔒 评估数据集与训练集严格分离

> **机器学习评估的基本原则**：评估集数据**绝不进入**训练集

### 实现机制

`download_all_data.py` 的 `create_eval_set()`：

1. 训练集先从原始数据集采样
2. 从训练集中**真正分离**评估样本（不是简单采样）
3. 写评估集到 `dataset/eval/eval_*.parquet`
4. 写回训练集到 `dataset/edu_*.parquet`（**已移除评估样本**）
5. 每个样本标记 `split='train'/'eval'`

### 验证

```bash
python -c "
import pyarrow.parquet as pq
t = pq.read_table('dataset/edu_science.parquet').to_pylist()
e = pq.read_table('dataset/eval/eval_science.parquet').to_pylist()
overlap = set(r.get('question','')[:100] for r in t) & set(r.get('question','')[:100] for r in e)
print(f'重叠: {len(overlap)} (应为 0)')"
```

### 旧版本升级

```bash
Remove-Item dataset\edu_*.parquet, dataset\eval\eval_*.parquet
python download_all_data.py
```

---

## 📚 相关文档

| 文档 | 内容 |
|------|------|
| **[DATA.md](DATA.md)** | 22 个数据集来源 / 许可 / 特点 / 下载方式 |
| **[EVAL_DESIGN.md](EVAL_DESIGN.md)** | 评估系统设计（19 评估集 + 4 阶段 + 8 工具） |

---

## 📜 License

Apache 2.0 | 基座模型 [Qwen2-VL](https://github.com/QwenLM/Qwen-VL) 遵循其原始协议

## 🙏 致谢

- [MiniMind-V](https://github.com/jingyaogong/minimind-v) — 项目架构与 README 风格参考
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) — 基座多模态模型
- 22 个开源数据集作者（详见 [DATA.md](DATA.md)）
