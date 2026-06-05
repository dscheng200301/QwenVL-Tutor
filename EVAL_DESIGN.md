# QwenSearch 评估系统设计文档

## 一、概述

评估系统是 QwenSearch 训练管线的关键环节，承担三大职责：

1. **能力度量** — 量化模型在 12 个数据集上的解题能力
2. **退化检测** — 追踪各训练阶段是否破坏了已有能力
3. **反馈优化** — 评估结果指导数据重采样，实现"弱项多练"

## 二、评估矩阵

### 2.1 六阶段评估

| 阶段 | 命令 | 评估内容 | 目标 |
|------|------|----------|------|
| **baseline** | `--stage baseline` | ScienceQA + C-Eval 基线 | 建立训练前的参照基准 |
| **sft** | `--stage sft` | ScienceQA + C-Eval + 自定义数据集 | 验证监督微调效果 |
| **dpo** | `--stage dpo` | ScienceQA + 偏好 gap + 退化检测 | 确认偏好对齐不破坏基础能力 |
| **grpo** | `--stage grpo` | ScienceQA + 奖励质量 + 退化检测 | 确认强化优化正向提升 |
| **full** | `--stage full` | ScienceQA test split (全量 4,241) + C-Eval holdout (500) | 最终发布前的全量验证 |
| **fine** | `--stage fine` | 五维度细粒度评分 | 诊断各维度能力短板 |

### 2.2 评估数据集（12 个）

| 数据集 | 评估样本数 | 来源 | 评估指标 |
|--------|-----------|------|----------|
| ScienceQA | 932 | 训练集 15% holdout | 答案准确率 + 步骤完整率 + 启发式引导率 |
| C-Eval | 500 | 5 个保留 HF 学科 | 选项匹配率 |
| OCR-VQA | 1,000 | 训练集 15% holdout | 关键词匹配率 |
| Ape210K | 1,000 | 训练集 15% holdout | 关键词匹配率 |
| OpenR1-Math | 1,000 | 训练集 15% holdout | 关键词匹配率 + 步骤完整率 |
| ChartQA | 1,000 | 训练集 15% holdout | 答案匹配率 |
| CMMLU | 1,000 | 训练集 15% holdout | 选项匹配率 |
| MathVerse | 591 | 训练集 15% holdout | 关键词匹配率 |
| MathVista | 150 | 训练集 15% holdout | 答案匹配率 |
| RACE | 1,000 | 训练集 15% holdout | 选项匹配率 |
| Gaokao MathQA | 351 | 全量 (小数据集) | 选项匹配率 |
| Gaokao MathCloze | 118 | 全量 (小数据集) | 数值匹配率 |

### 2.3 评估指标

| 指标 | 计算方法 | 适用场景 |
|------|----------|----------|
| **答案准确率** | `GT答案 ⊂ 模型回复` | ScienceQA, MathVista, Gaokao |
| **选项匹配率** | 选项字母是否出现 | C-Eval, CMMLU, RACE |
| **关键词匹配率** | GT 关键词 ∩ 回复关键词 / GT 关键词 | OCR-VQA, Ape210K, MathVerse, OpenR1-Math |
| **步骤完整率** | 回复是否包含"步骤/首先/然后/Step" | 所有数学推理数据集 |
| **启发式引导率** | 回复是否包含"观察/思考/想一想/你能发现" | 亲子教育场景评估 |

### 2.4 五维度细粒度评分（`--stage fine`）

| 维度 | 权重 | 评分逻辑 |
|------|:----:|----------|
| 答案准确性 | 30% | GT 答案关键词与回复关键词的重叠率 (Jaccard) |
| 步骤完整性 | 25% | 包含"第一步/第二步/步骤/Step"等推理结构词的数量 |
| 语言流畅度 | 15% | 中文/英文句子长度适中，无重复碎片语 |
| 启发式引导 | 20% | 包含"观察/思考/想一想/试试看/为什么"等引导词 |
| 格式规范性 | 10% | 答案位置是否清晰（如"答案是"、"最终答案"） |

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

## 四、评估反馈优化闭环

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  训练     │ →  │  评估     │ →  │  对比     │
│ train_*.py│     │eval_edu.py│     │compare_.py│
└──────────┘     └──────────┘     └──────────┘
                      ↓                 ↓
               eval_results/     指标变化报告
               <stage>_<ts>.json  (✅上升/⚠️下降)
                      ↓
               ┌──────────┐
               │  重采样   │
               │resample_.py│
               └──────────┘
                      ↓
                弱项数据集权重 ↑
                强项数据集权重 ↓
                      ↓
               ┌──────────┐
               │  重新训练  │ ───→ 循环
               └──────────┘
```

### 4.1 步骤 1：评估 + 持久化

```bash
python eval_edu.py --model_path out/edu_sft --stage sft --eval_all --max_samples 200
```

结果自动保存到 `eval_results/sft_20250605_143000.json`，内容结构：

```json
{
  "timestamp": "20250605_143000",
  "stage": "sft",
  "model_path": "out/edu_sft",
  "accuracy": 0.723,
  "step_completeness": 0.891,
  "scaffolding_rate": 0.456,
  "dataset": "ScienceQA-validation",
  "total": 932
}
```

同时更新 `eval_results/latest.json` 追踪最新结果。

### 4.2 步骤 2：对比分析

```bash
python compare_evals.py
```

自动找到最新的两个评估文件，逐指标对比：

```
📊 评估结果对比分析
  指标                    文件1      文件2        变化      趋势
  accuracy                0.6123     0.7230     +0.1107    ✅ 上升
  step_completeness       0.7800     0.8910     +0.1110    ✅ 上升
  scaffolding_rate        0.3400     0.4560     +0.1160    ✅ 上升
```

也可手动指定文件对比：

```bash
python compare_evals.py --file1 eval_results/sft_20250601.json --file2 eval_results/sft_20250605.json
```

### 4.3 步骤 3：数据重采样

```bash
python scripts/resample_data.py
```

**核心公式：**

```
调整系数 = 0.5 / max(数据集得分, 0.01)
最终权重 = clamp(调整系数, 0.3, 3.0)
```

**设计原理：**

- 得分低于 50% 的数据集 → 权重 > 1.0（弱项加强训练）
- 得分高于 50% 的数据集 → 权重 < 1.0（强项适当减少）
- 权重下限 0.3（避免完全放弃某个数据集）
- 权重上限 3.0（避免某个数据集完全主导训练）

**输出示例：**

```
📊 基于评估反馈的重采样权重
  数据集                 得分      权重      建议
  cmmlu                 0.320     1.56      🟡 加强训练
  chartqa               0.450     1.11      
  scienceqa             0.780     0.64      
  gaokao_mathqa         0.210     2.38      🔴 重点训练
```

同时输出推荐的训练命令：

```bash
python trainer/train_sft.py --data_paths "dataset/edu_science.parquet,dataset/edu_ceval.parquet,..."
```

可通过 `--output weights.json` 导出权重配置供后续使用。

## 五、代码架构

```
eval_edu.py              # 主评估脚本
├── load_model()         # 加载 QwenSearchVLM + LoRA
├── evaluate_scienceqa() # ScienceQA 评估（HF validation/test split）
├── evaluate_ceval()     # C-Eval 中文理科评估（5学科×HF streaming）
├── evaluate_custom()    # 本地 Parquet 数据集通用评估
├── evaluate_dpo_quality()    # DPO 偏好 gap 评估（chosen vs rejected）
├── evaluate_grpo_reward()    # GRPO 奖励质量评估
├── evaluate_regression()     # 退化检测（对比 baseline.json）
├── evaluate_fine_grained()   # 五维度细粒度评分
├── save_eval_results()       # 结果持久化
└── CLI (argparse)            # 6 个 stage + --eval_all + --eval_dataset

compare_evals.py         # 结果对比工具
├── find_latest_files()       # 自动找最新两个评估文件
└── compare_results()         # 逐指标 diff + 趋势判定

scripts/resample_data.py # 数据重采样工具
├── find_latest_eval()        # 自动找最新评估结果
├── compute_resample_weights() # 权重计算（弱项多练公式）
└── generate_data_paths()     # 生成 --data_paths 参数字符串
```

## 六、典型工作流

### 6.1 首次训练流程

```bash
# 1. 训练前打基线
python eval_edu.py --stage baseline --max_samples 500
# → 保存 eval_results/baseline.json + eval_results/baseline_<ts>.json

# 2. SFT 训练
python trainer/train_sft.py --epochs 3 --save_weight edu_sft

# 3. 全量评估
python eval_edu.py --model_path out/edu_sft --stage sft --eval_all --max_samples 300

# 4. 对比 vs 基线（检查提升幅度）
python compare_evals.py
```

### 6.2 迭代优化流程

```bash
# 1. 训练
python trainer/train_sft.py --epochs 3

# 2. 评估所有数据集
python eval_edu.py --model_path out/edu_sft --stage sft --eval_all --max_samples 200

# 3. 分析弱项
python compare_evals.py

# 4. 生成数据重采样建议
python scripts/resample_data.py

# 5. 使用建议的 --data_paths 重新训练
python trainer/train_sft.py --data_paths "..." --epochs 2
```

### 6.3 DPO/GRPO 安全验证流程

```bash
# DPO 训练后
python eval_edu.py --stage dpo --model_path out/edu_dpo --max_samples 100
# → 自动执行偏好 gap 评估 + 退化检测
# → 若退化 >10%，脚本警告并建议回退

# GRPO 训练后
python eval_edu.py --stage grpo --model_path out/edu_grpo --max_samples 100
# → 自动执行奖励质量评估 + 退化检测
```

## 七、与训练脚本的集成

### 7.1 训练前基线检查提示

训练脚本 (`train_sft.py`, `train_dpo.py`, `train_grpo.py`) 启动时建议检查基线是否存在：

```python
if not os.path.exists("eval_results/baseline.json"):
    print("⚠️ 未找到评估基线，建议先运行: python eval_edu.py --stage baseline")
```

### 7.2 退化触发训练中断

如果 DPO/GRPO 阶段退化检测返回严重退化（准确率下降 >10%），训练者可手动中止当前训练并在下次启动时降低 lr。

## 八、注意事项

1. **基线必须存在** — 退化检测依赖 `eval_results/baseline.json`，首次训练前务必运行 `--stage baseline`
2. **随机种子固定** — 评估使用 `seed=42`，但采样温度不为 0 时结果仍有波动
3. **GaoKao 数据集不全量隔离** — 351/118 条数据量太少，训练和评估共享同一份数据（合理取舍）
4. **C-Eval 评估从 HF 实时加载** — 需要网络连接，首次运行会下载 5 个保留学科的 test split
5. **`--eval_all` 耗时较长** — 评估 12 个数据集 × 200 条约需 30-60 分钟（取决于 GPU）