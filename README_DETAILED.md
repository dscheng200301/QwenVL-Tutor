# QwenVL-Tutor 详细文档

> 本文档为 [README.md](README.md) 的补充，包含项目架构设计、技术细节和高级用法。快速上手请先阅读 [README.md](README.md)。

---

## 核心特性

- **22 个训练数据集**：~222K 条样本，中文图文占比 **52.2%**
- **两阶段训练**：SFT → GRPO（移除 DPO，避免错误偏好信号）
- **19 个评估数据集**：Bootstrap 置信区间 + p-value 显著性检验
- **一站式工具**：`edu_evaluate.py all` + `edu_optimize.py auto`
- **严格数据分离**：评估集数据**绝不进入**训练集

---

## 硬件要求

| 配置 | 最低 | 推荐 | 最佳 |
|------|------|------|------|
| GPU | RTX 3090 (24GB) | A100 (40GB) | A100 (80GB) / H100 |
| 显存 | 24 GB | 40 GB | 80 GB |
| 内存 | 32 GB | 64 GB | 128 GB |
| 硬盘 | 100 GB SSD | 200 GB NVMe | 500 GB NVMe |

### 预估时间（完整流程）

| GPU | SFT (3 ep) | GRPO (1 ep) | 评估 (19x200) | 总计 |
|-----|:----------:|:-----------:|:-------------:|:----:|
| RTX 3090 | 36-48 h | 6-10 h | 3-5 h | **50-70 h** |
| A100 40GB | 12-18 h | 2-4 h | 1-2 h | **18-28 h** |
| A100 80GB | 8-12 h | 1-2 h | 1 h | **12-18 h** |

> 配置：`batch_size=4 x grad_accum=8`（有效 batch=32），`max_seq_len=2048`

---

## 核心依赖

| 类别 | 包 |
|------|---|
| 核心 ML | torch, transformers, accelerate, peft |
| 数据 | datasets, pyarrow, Pillow, numpy |
| 分布式训练（可选） | deepspeed |
| 推理加速（可选） | vllm |
| 监控 | wandb, swanlab |
| 演示 | gradio |

> vLLM 不可用时 `edu_evaluate.py` 会自动降级到 HuggingFace transformers

---

## 训练管线（SFT → GRPO 两阶段）

```
基模评估 (edu_evaluate.py run --stage baseline) → 建立基线
    → SFT 训练 (22 datasets, ~222K, weighted sampling)
    → SFT 评估 (edu_evaluate.py all)  → 自动生成 report.md
    → SFT 优化 (edu_optimize.py auto) → resample + retrain
    → GRPO 训练 (5K curated, 5 维度奖励)  → 从 SFT 衔接
    → GRPO 评估 (edu_evaluate.py all --stage grpo)
    → GRPO 优化 (edu_optimize.py grpo) → 自动决策：调整超参 / 回退 SFT
    → 最终评估 → 19 datasets + ScienceQA test split
```

**为什么去掉 DPO？** 当前 DPO 偏好数据用「简单截断」构造，本质是长度偏好而非质量偏好，训练风险大于收益。GRPO 的奖励模型已能直接优化「引导性」维度。

---

## 数据集概览

| 类别 | 数量 | 代表数据集 | 训练 / 评估 |
|------|:----:|-----------|:-----------:|
| 中文核心图文数学 | 3 | We-Math 2.0, Geo170K, windata-math | ~70K / 1.5K |
| 中文多学科图文做题 | 4 | CMMU, CMMMU, M3Exam, MMSciBench | ~8.8K / 0.7K |
| 核心图文数学 | 3 | ScienceQA, MathVerse, MathVista | ~11K / 1.7K |
| OCR / 图表 | 2 | OCR-VQA, ChartQA | ~30K / 2K |
| 中文理科 / 数学 | 5 | C-Eval, CMMLU, Ape210K, OpenR1-Math, Gaokao | ~55K / 3K |
| 语言理解 | 1 | RACE | 10K / 1K |
| **合计** | **22** | — | **~222K / 8,950** |

详细说明（来源、许可、下载）见 **[DATA.md](DATA.md)**。

---

## GRPO 奖励模型

### 规则化奖励模型（EduRewardModel）

位于 `trainer/reward_model.py`。五维度评分（总分 0~1），无需训练 NN：

| 维度 | 权重 | 计算方法 |
|------|:----:|----------|
| 答案准确性 | 0.30 | 关键词匹配 + TF-IDF 语义相似度 + 数值容差 |
| 步骤完整性 | 0.25 | 解题步骤结构词（"第一步"、"然后"、"所以"等） |
| 启发式引导 | 0.20 | 引导词检测 - 直接给答案的负面模式 |
| 语言流畅度 | 0.15 | 中英文表达流畅度 |
| 格式规范性 | 0.10 | "答案是 X"、"最终答案为 X" 等格式 |

**配套函数：**
- `EduRewardModel.compute_reward(response, gt)` → 单条 reward
- `EduRewardModel.compute_group_rewards(responses, gt)` → 组内 K 个候选的 rewards
- `edu_grpo_advantage(rewards)` → 组内标准化优势
- `edu_grpo_policy_loss(...)` → GRPO 策略损失（含 KL 散度）

**核心设计：**
```python
# 引导关键词（正面）
SCAFFOLDING_KEYWORDS = ["观察", "思考", "想一想", "你能发现", ...]

# 直接给答案（负面）
DIRECT_ANSWER_PATTERNS = [r"^[A-D][\.\)]\s*\n", r"^答案[是为]：[A-D]", ...]

# 数字同义词映射
NUMBER_KEYWORDS = {"一": "1", "二": "2", "两": "2", "加倍": "x2", ...}
```

特点：规则化（不训练 NN）、轻量级（TF-IDF 相似度无 GPU 开销）、中文优化、教育领域专用。

### LLM-as-Judge 奖励模型（推荐）

位于 `trainer/llm_reward.py`。当前 GRPO 训练默认使用 **API 后端**（`APILLMRewardModel`），准确率 85-92%，远超规则模型（70-75%）。

#### 4 种后端对比

| 后端 | 成本 | 速度 | 质量 | 硬件 |
|------|------|------|------|------|
| `APILLMRewardModel` | 付费 | 中 | 最高 | 任意 |
| `LocalVLLMRewardModel` | 0（一次性） | 快 | 高 | 24GB+ 显存 |
| `LocalHFRewardModel` | 0 | 慢 | 高 | 任意 |
| `HybridLLMRewardModel` | 0/付费 | 中 | 最高 | 任意 |

#### 评分维度

| 维度 | 权重 | 说明 |
|------|:----:|------|
| 答案准确性 | 0.40 | 与标准答案一致性（LLM 严格比对） |
| 步骤完整性 | 0.20 | 是否给出清晰步骤 |
| 启发式引导 | 0.20 | 引导式提问 vs 直接给答案 |
| 语言流畅度 | 0.10 | 中文表达质量 |
| 格式规范性 | 0.10 | "答案是 X" 格式 |

#### 使用方式

```bash
# 默认：API 后端（需要 OPENAI_API_KEY）
python trainer/train_grpo.py --from_weight ../out/edu_sft --api_model gpt-4o-mini

# 使用 DeepSeek 等兼容 API
python trainer/train_grpo.py --from_weight ../out/edu_sft \
    --api_model deepseek-chat \
    --api_base_url https://api.deepseek.com/v1 \
    --api_key sk-xxx
```

```python
# 代码中手动选择后端
from trainer.llm_reward import create_llm_reward

# 自动选择（优先 vLLM > HF > API）
reward = create_llm_reward(backend="auto")

# 本地 vLLM（推荐，0 成本）
reward = create_llm_reward(
    backend="vllm",
    model_path="Qwen/Qwen2.5-72B-Instruct-AWQ",
    tensor_parallel_size=1,
)

# 混合（防 reward hacking）
from trainer.reward_model import EduRewardModel
reward = create_llm_reward(
    backend="hybrid",
    llm_model=create_llm_reward(backend="vllm"),
    rule_model=EduRewardModel(),
)
```

#### 实施建议

1. **短期**：先用 API 后端跑通流程（小规模验证）
2. **中期**：用 `LocalVLLMRewardModel` 替代（成本 0，质量大幅提升）
3. **长期**：用 `HybridLLMRewardModel`（防 reward hacking，最稳健）

> **GRPO 中 LLM 必须冻结**：训练时只用 LLM 评分，**不能让 LLM 一起更新**，否则 reward 会偏向自己。LLM 在训练中仅作为「评分员」。

---

## 终端实时可视化

位于 `trainer/terminal_dashboard.py`。训练/评估/优化过程自动显示**实时进度窗口**：进度条、loss 曲线、GPU 显存、ETA。无需手动配置，基于 rich 库（已在 `requirements.txt`）。

### 训练时

```
┌─ SFT Training ──────────────────────────────────────┐
│ ▶ Training ████████▌░░░░░░░░  156/200  ● 02:30  ● ETA 01:00 │
│                                                       │
│ 📊 实时指标                                            │
│   train/loss          1.2345  ↓ -0.0234              │
│   train/learning_rate  0.0001  → 平稳                 │
│   train/throughput     4.23    ↑ +0.10               │
│                                                       │
│ 💻 系统状态                                            │
│   GPU 0 显存    20.5 / 40.0 GB (51%)                 │
│   已用时         150 秒                               │
└───────────────────────────────────────────────────────┘
```

### 评估时

```
Evaluation 开始，共 19 个数据集

✔ scienceqa (200 样本)
  [████████░░░░░░░░░░] 156/200 (78%) acc=0.6123
✔ math_vista (150 样本)
  [████████░░░░░░░░░░] 142/150 (94%) acc=0.5342
...

✅ Evaluation 完成
┌─ 评估结果汇总 ──────────────┐
│ 数据集      准确率  样本数  │
│ scienceqa  0.6123   200    │
│ math_vista 0.5342   150    │
│ ...                       │
│ 平均       0.5823   —      │
└────────────────────────────┘
```

### 关闭（如不需要）

```bash
QwenVL-Tutor_NO_DASHBOARD=1 python trainer/train_sft.py --epochs 3
QwenVL-Tutor_NO_DASHBOARD=1 python scripts/eval/edu_evaluate.py all ...
```

> 不依赖 wandb，离线运行也可用。

---

## wandb 训练监控

用 wandb 记录训练/评估/优化全过程的曲线，便于对比和调参。首次使用需 `wandb login`；国内网络可用 `swanlab` 替代（`pip install swanlab`）。

### 启用 wandb

```bash
# 训练时启用
python trainer/train_sft.py --use_wandb --wandb_project QwenVL-Tutor
python trainer/train_grpo.py --use_wandb --wandb_project QwenVL-Tutor

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
wandb login
wandb sweep / WANDB_PROJECT=QwenVL-Tutor wandb graph
wandb compare --project QwenVL-Tutor
```

> 训练脚本已内置 wandb 集成，启用后无需额外配置。GPU 监控建议用 `nvidia-smi dmon` 或 wandb 的 system metrics。

---

## 评估数据集与训练集严格分离

> **机器学习评估的基本原则**：评估集数据**绝不进入**训练集。

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

## 评估报告命名

每次评估生成的报告**自动命名**为 `eval_results/{stage}_{YYYYMMDD}_{HHMM}_acc{X.XXXX}.md`：

```
eval_results/
├── sft_20260605_1430_acc0.6123.md       # SFT 后评估，平均准确率 0.6123
├── grpo_20260605_1530_acc0.6845.md      # GRPO 强化后评估
├── full_20260606_0900_acc0.6912.md      # 最终发布评估
└── compare_20260606_1000_acc0.6500.md   # 对比评估
```

**手动指定**（可选）：

```bash
python scripts/eval/edu_evaluate.py report --output my_report.md
```

---

## 相关文档

| 文档 | 内容 |
|------|------|
| **[README.md](README.md)** | 项目概述与快速开始 |
| **[DATA.md](DATA.md)** | 22 个数据集来源 / 许可 / 特点 / 下载方式 |
| **[EVAL_DESIGN.md](EVAL_DESIGN.md)** | 评估系统设计（19 评估集 + 4 阶段 + 工具） |

---

## License

Apache 2.0 | 基座模型 [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) 遵循其原始协议

## 致谢

- [MiniMind-V](https://github.com/jingyaogong/minimind-v) — 项目架构与 README 风格参考
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) — 基座多模态模型
- 22 个开源数据集作者（详见 [DATA.md](DATA.md)）