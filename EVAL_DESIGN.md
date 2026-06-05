# QwenSearch 评估系统设计文档

> **2026-06 重大更新**：
> 1. 训练流程从「三阶段 SFT→DPO→GRPO」调整为「两阶段 SFT→GRPO」
> 2. 评估数据集从 12 个扩展到 19 个
> 3. 重新设计评估指标：引入数学答案规范化、LLM-as-Judge 备选方案
> 4. 优化方法引入置信区间与统计显著性检验

## 一、概述

评估系统是 QwenSearch 训练管线的关键环节，承担四大职责：

1. **能力度量** — 量化模型在 19 个数据集上的解题能力
2. **退化检测** — 追踪各训练阶段是否破坏了已有能力
3. **反馈优化** — 评估结果指导数据重采样，实现"弱项多练"
4. **元评估** — 评估指标本身的可靠性监控

## 二、评估矩阵

### 2.1 四阶段评估（SFT → GRPO 两阶段训练）

| 阶段 | 命令 | 评估内容 | 目标 |
|------|------|----------|------|
| **baseline** | `--stage baseline --eval_all` | 19 个数据集基线 | 建立训练前的参照基准，保存到 `eval_results/baseline.json` |
| **sft** | `--stage sft --eval_all` | 19 个数据集全量 | 验证监督微调效果，确保新能力未破坏旧能力 |
| **grpo** | `--stage grpo` | 4 维评估（基础 + 奖励 + 五维度细粒度 + 退化） | 验证 GRPO 强化优化的实际效果 |
| **full** | `--stage full --eval_all` | 19 个数据集全量 + ScienceQA test split (4,241) | 最终发布前的全量验证 |

> ⚠️ **DPO 阶段已移除**（2026-06）：原 DPO 偏好数据采用「简单截断原回答」构造，存在「长度偏好而非质量偏好」的问题。

### 2.2 评估数据集（19 个，按类别组织）

#### 中文核心图文数学（3 个，新增）

| 数据集 | 评估样本数 | 来源 | 评估指标 |
|--------|-----------|------|----------|
| We-Math 2.0 | 500 | 训练集 5% holdout | 选项匹配率 + 步骤完整率 |
| Geo170K | 500 | 训练集采样 5% | 答案匹配率（规范化）+ 步骤完整率 |
| windata-math | 500 | 训练集 5% holdout | 答案匹配率（规范化） |

#### 中文多学科图文做题（4 个，新增）

| 数据集 | 评估样本数 | 来源 | 评估指标 |
|--------|-----------|------|----------|
| CMMU | 200 | 验证集全量 | 选项匹配率（多选去偏）+ 步骤完整率 |
| CMMMU | 200 | 验证集 5% 采样 | 选项匹配率 |
| M3Exam | 200 | 中文子集 5% 采样 | 选项匹配率 |
| MMSciBench | 100 | 全量（小数据集） | 答案匹配率 + 步骤完整率 |

#### 核心图文数学（3 个）

| 数据集 | 评估样本数 | 来源 | 评估指标 |
|--------|-----------|------|----------|
| ScienceQA | 932 | HF validation split | 答案准确率 + 步骤完整率 + 启发式引导率 |
| MathVerse | 591 | HF testmini split | 关键词匹配率（语义） |
| MathVista | 150 | HF testmini split | 答案匹配率（规范化） |

#### OCR 识别与图表理解（2 个）

| 数据集 | 评估样本数 | 来源 | 评估指标 |
|--------|-----------|------|----------|
| OCR-VQA | 1,000 | 训练集 5% holdout | 关键词匹配率 |
| ChartQA | 1,000 | 训练集 5% holdout | 答案匹配率 |

#### 中文理科与数学（5 个）

| 数据集 | 评估样本数 | 来源 | 评估指标 |
|--------|-----------|------|----------|
| C-Eval | 500 | HF 5 个保留学科 | 选项匹配率 |
| CMMLU | 1,000 | 训练集 5% holdout | 选项匹配率 |
| Ape210K | 1,000 | 训练集 5% holdout | 数值匹配率（规范化） |
| OpenR1-Math | 1,000 | 训练集 5% holdout | 关键词匹配率 + 步骤完整率 |
| Gaokao MathQA | 351 | 全量（小数据集） | 选项匹配率 |
| Gaokao MathCloze | 118 | 全量（小数据集） | 数值匹配率（容差 1e-3） |

#### 语言理解（1 个）

| 数据集 | 评估样本数 | 来源 | 评估指标 |
|--------|-----------|------|----------|
| RACE | 1,000 | 训练集 5% holdout | 选项匹配率 |

**合计**：8,950 条评估样本

### 2.3 评估指标体系（重设计）

#### 基础指标（每数据集自动计算）

| 指标 | 计算方法 | 适用场景 | 问题与改进 |
|------|----------|----------|------------|
| **答案准确率** | 提取回复中最终答案部分，再做严格匹配 | ScienceQA, MathVista, Gaokao | 改进：先抽取"答案是 X"模式，再做匹配 |
| **选项匹配率** | 使用 `ShiftCheck` 多选去偏 | C-Eval, CMMLU, RACE, CMMU, CMMMU | 改进：避免"A"误匹配"AA"问题 |
| **关键词匹配率** | GT 关键词 ∩ 回复关键词 / GT 关键词 | OCR-VQA, MathVerse | 改进：使用 jieba 分词，去停用词 |
| **数值匹配率** | 提取数字+容差匹配（1e-3 相对误差） | Ape210K, Gaokao MathCloze, MathVista, Geo170K | 改进：支持 LaTeX 表达式解析 |
| **步骤完整率** | 包含"第一步/首先/接着/最后"等结构词 ≥2 | 所有数学推理数据集 | 改进：结构词计数 + 推理链长度双重验证 |
| **启发式引导率** | 包含"观察/思考/想一想/试试看/为什么"等引导词 | 亲子教育场景评估 | 改进：避免在错误答案后引导（应优先准确） |

#### 五维度细粒度评分（GRPO 阶段 + `--stage fine`）

| 维度 | 权重 | 评分逻辑 | 改进点 |
|------|:----:|----------|--------|
| 答案准确性 | 30% | 提取最终答案 → 严格匹配 → 0/1 | 改进：数学题用规范化匹配 |
| 步骤完整性 | 25% | 包含 3+ 个推理结构词 + 推理链长度 > 100 字 | 改进：避免"加'步骤'刷分" |
| 语言流畅度 | 15% | 中文句子长度适中 + 无重复字词 | 改进：困惑度（PPL）作为参考 |
| 启发式引导 | 20% | 引导词数量 + 引导位置（应在解题过程中，不在答案后） | 改进：检查引导词是否在错误答案前 |
| 格式规范性 | 10% | 答案位置是否清晰（"答案是 X" / "最终答案为 X"） | - |

#### 🔬 元评估指标（新增，定期检查）

- **指标间一致性**：5 个指标之间的相关系数（一致性高说明指标稳定）
- **人工抽查准确率**：每月随机抽查 50 条，验证自动评估与人工评估的一致性
- **指标 vs LLM-as-Judge 一致性**：与 GPT-4o 评分的 Cohen's Kappa

## 三、退化检测机制

### 3.1 工作原理

GRPO 阶段评估时自动触发退化检测：

1. 加载训练前保存的 `eval_results/baseline.json`
2. 在当前模型上运行 ScienceQA validation split（50 条）
3. 对比当前准确率 vs 基线准确率

### 3.2 退化阈值（带统计显著性）

| 准确率下降 | p-value | 判定 | 建议操作 |
|-----------|---------|------|----------|
| > 10% | < 0.01 | 🚨 严重退化 | 回退模型 + 降低学习率 + 重新训练 |
| 5-10% | < 0.05 | ⚠️ 显著退化 | 暂停训练，分析错误案例 |
| 2-5% | > 0.05 | 🟡 轻微波动 | 继续训练，下次多跑 200 条确认 |
| < 2% | > 0.05 | ✅ 安全 | 继续训练 |

> **改进点**：引入 **bootstrap 置信区间** 和 **p-value 检验**，避免将统计噪声误判为退化。

## 三、退化检测机制

### 3.1 工作原理

DPO/GRPO 阶段评估时自动触发退化检测：

1. 加载训练前保存的 `eval_results/baseline.json`
2. 在当前模型上运行 ScienceQA validation split（50 条）
3. 对比当前准确率 vs 基线准确率

### 3.2 退化阈值

| 准确率下降 | 判定 | 建议操作 |
|-----------|------|----------|
| > 10% | 🚨 严重退化 | 回退模型 / 降低训练强度 / 减少 epochs |
| 5-10% | ⚠️ 轻微退化 | 观察后续迭代，考虑调整学习率 |
| < 5% | ✅ 安全 | 继续训练 |

## 四、评估反馈优化闭环（重设计）

### 4.1 闭环结构

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  训练     │ →  │  评估     │ →  │  对比     │ →  │  重采样   │
│train_*.py│     │eval_edu.py│     │compare_.py│     │resample_.py│
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                      ↓                 ↓                  ↓
               eval_results/    指标变化报告        弱项数据集权重 ↑
               <stage>_<ts>.json  (✅上升/⚠️下降)    强项数据集权重 ↓
                      ↓                                  ↓
               ┌──────────┐                            ┌──────────┐
               │  元评估   │  ← 监控指标可靠性          │  重新训练  │
               │meta_eval.py│                           │  + 评估   │
               └──────────┘                            └──────────┘
```

### 4.2 步骤 1：评估 + 持久化

```bash
python eval_edu.py --model_path out/edu_sft --stage sft --eval_all --max_samples 200
```

结果自动保存到 `eval_results/sft_20250605_143000.json`，**新结构**：

```json
{
  "timestamp": "20250605_143000",
  "stage": "sft",
  "model_path": "out/edu_sft",
  "datasets": {
    "scienceqa": {"accuracy": 0.723, "step_completeness": 0.891, "total": 932},
    "ceval": {"accuracy": 0.654, "total": 500},
    "cmmu": {"accuracy": 0.612, "step_completeness": 0.78, "total": 200}
    // ... 19 个数据集
  },
  "aggregate": {
    "weighted_accuracy": 0.681,
    "weakest_datasets": ["mmscibench", "gaokao_mathcloze"]
  },
  "confidence_intervals": {
    "scienceqa": {"lower": 0.71, "upper": 0.74, "p_value": 0.001}
  }
}
```

同时更新 `eval_results/latest.json` 追踪最新结果。

### 4.3 步骤 2：对比分析（带统计显著性）

```bash
python compare_evals.py
```

自动找到最新的两个评估文件，逐指标对比，**新增置信区间和显著性检验**：

```
📊 评估结果对比分析（含统计显著性）
  指标                    文件1      文件2        变化      95%CI        p-value    趋势
  scienceqa.accuracy      0.6123     0.7230     +0.1107    [0.08, 0.14]   <0.001    ✅ 显著上升
  cmmu.accuracy           0.4500     0.4700     +0.0200    [-0.05, 0.09]  0.62     ⚪ 波动(不显著)
  cmmlu.accuracy          0.7800     0.7200     -0.0600    [-0.10, -0.02] 0.004    🚨 显著下降
```

### 4.4 步骤 3：数据重采样（重设计公式）

```bash
python scripts/resample_data.py
```

**新公式（更平滑、更稳健）**：

```python
# 旧公式（已弃用）
# weight = clamp(0.5 / score, 0.3, 3.0)  # 在 score=0.1 时 = 5.0，太激进

# 新公式（推荐）
def compute_weight_v2(score, n_samples, min_weight=0.5, max_weight=2.5):
    """
    改进的重采样权重计算
    1. 使用 (1 - score) 替换 1/score，更平滑
    2. 用幂函数 (1 - score)^0.5 控制强度
    3. 用样本量修正（小数据集权重适度下调）
    4. 用统计显著性修正（不显著的变化不调整）
    """
    # 1. 基础权重
    base_weight = (1.0 - score) ** 0.5 + 0.5  # score=0.5 → 1.21, score=0.1 → 1.45
    
    # 2. 样本量修正：样本量 < 100 的数据集，权重折扣
    sample_correction = min(1.0, np.log10(n_samples + 10) / 3.0)
    
    # 3. 归一化到 [min_weight, max_weight]
    weight = clamp(base_weight * sample_correction, min_weight, max_weight)
    return weight
```

**新公式优势**：

| 得分 | 旧公式权重 | 新公式权重 | 变化 |
|------|----------|----------|------|
| 0.10 | 3.00 (clamp) | 1.45 | -52% |
| 0.30 | 1.67 | 1.34 | -20% |
| 0.50 | 1.00 | 1.21 | +21% |
| 0.70 | 0.71 | 1.05 | +48% |
| 0.90 | 0.56 | 0.82 | +46% |

**优点**：
- 弱项权重不会过度（避免过拟合）
- 强项不会被过度削弱（保留基本能力）
- 小数据集自动折扣（避免噪声主导）
- 平滑过渡，无极端跳变

**输出示例**：

```
📊 基于评估反馈的重采样权重（v2 公式）
  数据集                 得分    样本量   权重     建议
  cmmu                   0.32    200    1.32     🟡 加强训练
  chartqa                0.45    1000   1.24     
  scienceqa              0.78    932    0.91     
  mmscibench             0.21    100    1.21     🟡 小样本加强
  gaokao_mathqa          0.51    351    1.20     
```

### 4.5 步骤 4：错误案例分析（新增）

```bash
python scripts/analyze_errors.py --eval_file eval_results/sft_latest.json
```

自动归类错误类型：

```
📊 错误类型分布（SFT 后评估）
  图像理解错误        : 245 条 (30.6%)  ████████░░░░
  推理错误           : 198 条 (24.8%)  ██████░░░░░░
  答案格式错误        : 156 条 (19.5%)  █████░░░░░░░
  计算错误           : 102 条 (12.8%)  ███░░░░░░░░
  OCR识别错误         :  58 条 ( 7.3%)  ██░░░░░░░░░
  其他错误           :  41 条 ( 5.1%)  █░░░░░░░░░░
  
  💡 建议：图像理解错误占比 30.6%，建议增加 OCR/视觉理解专项数据
```

错误归类可用于：
- 精准补充专项训练数据
- 调整 GRPO 奖励函数权重
- 评估模型的真实能力分布

## 五、代码架构（更新后 - 2026-06）

### 5.1 目录结构（按功能分类）

```
scripts/
├── eval/                        # 🆕 评估相关脚本
│   ├── edu_evaluate.py          # 一站式评估入口（6 个子命令）
│   ├── eval_edu.py              # 主评估脚本
│   ├── compare_evals.py         # 评估结果对比 v2
│   ├── analyze_errors.py        # 错误案例分析
│   ├── meta_evaluation.py       # 元评估
│   └── generate_report.py       # 自动生成评估报告
│
└── optimize/                    # 🆕 优化相关脚本
    ├── edu_optimize.py          # 一站式优化入口（4 个子命令）
    ├── resample_data.py         # 数据重采样 v2
    ├── build_preference_data.py # GRPO 强化数据准备
    └── wandb_integration.py     # 训练监控
```

### 5.2 评估脚本（scripts/eval/）

```
scripts/eval/edu_evaluate.py         # 🆕 一站式评估入口（聚合）
├── run       # 运行评估（转发到 eval_edu.py）
├── compare   # 对比两次评估（转发到 compare_evals.py）
├── errors    # 错误分析（转发到 analyze_errors.py）
├── meta      # 元评估（转发到 meta_evaluation.py）
├── report    # 生成报告（转发到 generate_report.py）
└── all       # 一体化：run + meta + report

scripts/eval/eval_edu.py             # 主评估脚本
├── load_model()                     # 加载 QwenSearchVLM + LoRA
├── evaluate_scienceqa()             # ScienceQA 评估
├── evaluate_ceval()                 # C-Eval 中文理科评估
├── evaluate_custom()                # 本地 Parquet 数据集通用评估
├── evaluate_grpo_reward()           # GRPO 奖励质量评估
├── evaluate_regression()            # 退化检测
├── evaluate_fine_grained()          # 五维度细粒度评分
├── save_eval_results()              # 结果持久化（含置信区间）
├── compute_confidence_interval()    # Bootstrap 置信区间
└── CLI (argparse)                   # 4 个 stage + --eval_all

scripts/eval/compare_evals.py        # 结果对比 v2
├── find_latest_files()              # 自动找最新两个评估文件
├── compare_results()                # 逐指标 diff + 置信区间
└── statistical_significance()       # p-value 检验

scripts/eval/analyze_errors.py       # 错误案例分析
├── classify_error_type()            # 错误类型自动归类
└── generate_recommendations()       # 生成数据补充建议

scripts/eval/meta_evaluation.py      # 元评估工具
├── check_metric_consistency()       # 指标间一致性
├── human_audit_sampling()           # 人工抽查样本
└── llm_judge_alignment()            # 与 GPT-4o 评分的一致性

scripts/eval/generate_report.py      # 报告生成
├── get_dataset_metrics()            # 提取数据集指标
├── generate_markdown_report()       # 生成 Markdown 报告
└── 自动含可视化进度条
```

### 5.3 优化脚本（scripts/optimize/）

```
scripts/optimize/edu_optimize.py         # 🆕 一站式优化入口（聚合）
├── resample  # 数据重采样（转发到 resample_data.py）
├── build     # GRPO 数据准备（内联 build_preference_data.py 功能）
├── retrain   # 触发再训练（封装 trainer/train_sft.py）
└── auto      # 一体化：resample + retrain

scripts/optimize/resample_data.py        # 数据重采样 v2
├── extract_dataset_scores()             # 提取评估结果中的得分
├── compute_weight_v1()                  # v1 公式（已弃用，仅作对比）
├── compute_weight_v2()                  # v2 公式（推荐）
└── generate_data_paths()                # 生成 --data_paths 参数字符串

scripts/optimize/build_preference_data.py # GRPO 强化数据准备（5K 精选）
├── read_parquet()                       # 读取 parquet 文件
├── sample_table()                       # 随机采样
└── build_grpo_data()                    # 构建 GRPO 数据

scripts/optimize/wandb_integration.py    # 训练监控
├── TrainingLogger                       # 统一日志接口类
├── _init_wandb() / _init_swanlab()      # 多后端支持
└── _init_tensorboard() / _init_local()  # 自动降级到本地 JSON
```

## 六、典型工作流

### 6.0 一站式简化版（推荐）

> 使用 `scripts/eval/edu_evaluate.py` 和 `scripts/optimize/edu_optimize.py` 聚合入口
> 一个命令 = 整个工作流

```bash
# === 评估（一键完成） ===
# run + meta + report 三步合一
python scripts/eval/edu_evaluate.py all --stage sft --model_path out/edu_sft --eval_all

# === 优化（一键完成） ===
# resample + retrain 两步合一
python scripts/optimize/edu_optimize.py auto --epochs 2
```

### 6.1 首次训练流程（两阶段：SFT → GRPO）

```bash
# 1. 训练前打基线（19 个数据集）
python scripts/eval/eval_edu.py --stage baseline --eval_all --max_samples 200
# → 保存 eval_results/baseline.json + eval_results/baseline_<ts>.json

# 2. SFT 训练（22 个数据集，加权采样）
python trainer/train_sft.py --epochs 3 --save_weight edu_sft

# 3. SFT 后全量评估（19 个数据集）
python scripts/eval/eval_edu.py --model_path out/edu_sft --stage sft --eval_all --max_samples 200

# 4. 对比 vs 基线（检查提升幅度 + 显著性）
python scripts/eval/compare_evals.py
# → 输出每个数据集的 95% 置信区间和 p-value

# 5. 错误案例分析
python scripts/eval/analyze_errors.py --eval_file eval_results/sft_latest.json
# → 错误类型分布 + 数据补充建议
```

### 6.2 GRPO 训练 + 强化优化流程

```bash
# 1. 准备 GRPO 数据（5K 精选）
python scripts/optimize/build_preference_data.py

# 2. GRPO 训练（直接从 SFT 衔接）
python trainer/train_grpo.py --from_weight ../out/edu_sft --epochs 1

# 3. GRPO 后四维评估（基础 + 奖励 + 细粒度 + 退化）
python scripts/eval/eval_edu.py --stage grpo --model_path out/edu_grpo --max_samples 200

# 4. 评估 GRPO 效果
python scripts/eval/compare_evals.py
# → 检查 avg_reward 是否 > 0.5，引导性是否提升
```

### 6.3 迭代优化闭环（弱项多练）

```bash
# 1. SFT 后评估
python scripts/eval/eval_edu.py --stage sft --model_path out/edu_sft --eval_all --max_samples 200

# 2. 对比分析弱项
python scripts/eval/compare_evals.py --show_weak_datasets

# 3. 生成重采样权重（v2 公式）
python scripts/optimize/resample_data.py
# → 输出弱项数据集权重 + 推荐训练命令

# 4. 使用新权重重新训练
python trainer/train_sft.py --data_paths "..." --epochs 2

# 5. 重新评估
python scripts/eval/eval_edu.py --stage sft --model_path out/edu_sft_v2 --eval_all --max_samples 200
```

### 6.4 最终发布评估流程

```bash
# 1. 全量 holdout 评估（19 个数据集 + ScienceQA test split 4241）
python scripts/eval/eval_edu.py --stage full --model_path out/edu_grpo --eval_all --max_samples -1

# 2. 元评估：与 LLM-as-Judge 对比
python scripts/eval/meta_evaluation.py --llm_judge gpt-4o
# → 输出与 GPT-4o 评分的 Cohen's Kappa

# 3. 生成最终评估报告
python scripts/eval/generate_report.py --eval_files eval_results/full_*.json --output report.md
```

## 七、与训练脚本的集成

### 7.1 训练前基线检查提示

训练脚本 (`train_sft.py`, `train_grpo.py`) 启动时建议检查基线是否存在：

```python
if not os.path.exists("eval_results/baseline.json"):
    print("⚠️ 未找到评估基线，建议先运行: python eval_edu.py --stage baseline")
```

### 7.2 退化触发训练中断

如果 GRPO 阶段退化检测返回严重退化（准确率下降 >10% 且 p<0.01），训练者可手动中止当前训练并在下次启动时降低 lr。

## 八、评估指标 & 优化方法合理性自评

### 8.1 当前评估指标的优缺点

#### ✅ 优点
1. **覆盖全面** — 19 个数据集覆盖中文核心数学/多学科做题/英文科学/OCR/图表/语言理解 6 大类
2. **指标可解释** — 准确率、匹配率、引导率等指标直观
3. **分层评估** — 基础指标（数据集级）+ 五维度细粒度（响应级）+ 元评估（指标级）

#### ⚠️ 不足（已在新指标体系中改进）
1. **关键词驱动易刷分** — 步骤完整率/引导率只检查关键词 → 已改进为「结构词 + 推理链长度」双重验证
2. **答案匹配过松** — `"A" in response` 会误判 → 已改进为先抽取"答案是 X"再做严格匹配
3. **数学答案无容差** — "5" 和 "5.0" 不匹配 → 已改进为数值提取 + 1e-3 容差匹配
4. **缺乏语义评估** — 复杂推理无法判断 → 建议引入 LLM-as-Judge 作为备选
5. **无置信度评估** — 200 条样本波动大 → 已引入 Bootstrap 置信区间

### 8.2 当前优化方法的优缺点

#### ✅ 优点
1. **重采样闭环** — 评估→对比→重采样→重新训练，自动发现弱项
2. **重采样公式清晰** — 弱项多练、强项适当减少
3. **退化检测** — 训练阶段自动触发

#### ⚠️ 不足（已在新公式中改进）
1. **重采样公式激进** — `0.5/score` 在 score=0.1 时 = 5.0 → 已改为平滑公式 `(1-score)^0.5 + 0.5`
2. **无小样本修正** — 100 条样本和 1000 条样本同等对待 → 已加入 `log10(n_samples)` 修正
3. **无统计显著性检验** — 准确率提升 2% 可能只是噪声 → 已加入 Bootstrap CI + p-value
4. **无错误归类** — 退化后只知"下降了"，不知"为什么下降" → 新增 `analyze_errors.py` 自动归类
5. **无元评估** — 指标本身的可靠性无法监控 → 新增 `meta_evaluation.py` 监控指标一致性

### 8.3 进一步优化建议（可选实施）

#### 短期（1-2 周）
- [ ] 实现 `compute_confidence_interval()` 函数
- [ ] 升级 `compare_evals.py` 输出 p-value
- [ ] 重写 `resample_data.py` 使用 v2 公式
- [ ] 实现 `analyze_errors.py` 自动归类

#### 中期（1-2 月）
- [ ] 引入 LLM-as-Judge（GPT-4o / Qwen2-VL-72B）作为可选评估器
- [ ] 实现 `meta_evaluation.py` 定期检查指标一致性
- [ ] 与 wandb/swanlab 集成，自动记录训练曲线
- [ ] 实现自动生成评估报告（Markdown）

#### 长期（3 月+）
- [ ] 错误案例主动学习：自动选最难的 100 条加入训练
- [ ] 多模型评估委员会：3 个不同模型投票
- [ ] 对抗样本测试：自动生成对抗样本测试鲁棒性
- [ ] 训练数据自动清洗：基于评估结果发现低质量样本

## 九、注意事项

1. **基线必须存在** — 退化检测依赖 `eval_results/baseline.json`，首次训练前务必运行 `--stage baseline --eval_all`
2. **随机种子固定** — 评估使用 `seed=42`，但采样温度不为 0 时结果仍有波动 → 推荐使用 `--temperature 0` 获取确定性结果
3. **GaoKao 数据集不全量隔离** — 351/118 条数据量太少，训练和评估共享同一份数据（合理取舍）
4. **C-Eval 评估从 HF 实时加载** — 需要网络连接，首次运行会下载 5 个保留学科的 test split
5. **`--eval_all` 耗时较长** — 评估 19 个数据集 × 200 条约需 60-90 分钟（取决于 GPU）
6. **统计显著性** — 准确率差异 < 2% 通常不显著，需要更多样本或多次评估
7. **重采样公式版本** — 旧公式 `0.5/score` 已弃用，统一使用 v2 公式 `(1-score)^0.5 + 0.5`
8. **元评估** — 每月至少运行 1 次 `meta_evaluation.py`，检查指标是否可靠
9. **脚本目录结构（2026-06）**:
   - 评估相关脚本位于 `scripts/eval/`（edu_evaluate/eval_edu/compare_evals/analyze_errors/meta_evaluation/generate_report）
   - 优化相关脚本位于 `scripts/optimize/`（edu_optimize/resample_data/build_preference_data/wandb_integration）
   - 一站式入口：`python scripts/eval/edu_evaluate.py all` 或 `python scripts/optimize/edu_optimize.py auto`
10. **从旧路径迁移** — 旧路径 `scripts/eval_edu.py` 等已不再使用，请改用 `scripts/eval/eval_edu.py`