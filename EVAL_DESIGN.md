# QwenSearch 评估系统设计

## 概述

19 个评估数据集，涵盖 6 大类题型。支持 Bootstrap 置信区间 + p-value 显著性检验，评估结果驱动数据重采样优化。

## 评估矩阵

| 阶段 | 命令 | 评估内容 |
|------|------|----------|
| **baseline** | `--stage baseline --eval_all` | 19 数据集基线 |
| **sft** | `--stage sft --eval_all` | SFT 后全量评估 |
| **grpo** | `--stage grpo` | GRPO 后 4 维评估（基础+奖励+细粒度+退化） |
| **full** | `--stage full --eval_all` | 最终发布评估 |

## 评估数据集（19 个）

| 类别 | 数量 | 评估样本 |
|------|:----:|:--------:|
| 中文核心图文数学 | 3 | 1.5K |
| 中文多学科图文做题 | 4 | 0.7K |
| 核心图文数学 | 3 | 1.7K |
| OCR / 图表 | 2 | 2K |
| 中文理科 / 数学 | 5 | 3K |
| 语言理解 | 1 | 1K |
| **合计** | **19** | **8,950** |

## 评估指标

### 基础指标

| 指标 | 适用 |
|------|------|
| 答案准确率 | 选择题（抽取"答案是 X"严格匹配） |
| 选项匹配率 | 多选题（字母出现 + 去偏） |
| 关键词匹配率 | 简答题（jieba 分词去停用词） |
| 数值匹配率 | 数学题（1e-3 容差，支持 LaTeX） |
| 步骤完整率 | 推理题（≥2 推理结构词） |
| 引导率 | 亲子场景（引导词检测） |

### 五维度细粒度评分（GRPO 阶段）

| 维度 | 权重 | 说明 |
|------|:----:|------|
| 答案准确性 | 0.30 | 提取最终答案 → 严格匹配 |
| 步骤完整性 | 0.25 | 3+ 结构词 + 推理链 > 100 字 |
| 启发式引导 | 0.20 | 引导词 + 位置正确 |
| 语言流畅度 | 0.15 | 句子长度适中 + 无重复 |
| 格式规范性 | 0.10 | "答案是 X" / "最终答案为 X" |

## 退化检测

| 准确率下降 | p-value | 判定 | 操作 |
|-----------|---------|------|------|
| > 10% | < 0.01 | 严重退化 | 回退 + 重训 |
| 5-10% | < 0.05 | 显著退化 | 暂停 + 分析 |
| 2-5% | > 0.05 | 轻微波动 | 继续训练 |
| < 2% | > 0.05 | 安全 | 继续训练 |

## 反馈闭环

```
训练 → 评估 (Bootstrap CI) → 错误归类 → 重采样 (v2 公式) → 重新训练
```

重采样公式 v2：`weight = clamp((1-score)^0.5 + 0.5, 0.5, 2.5) * log10(n+10)/3.0`

## 脚本架构

```
scripts/eval/
├── edu_evaluate.py          # 一站式入口（6 子命令）
│   ├── run       # 主评估
│   ├── compare   # 对比 + p-value
│   ├── errors    # 错误归类
│   ├── meta      # 指标一致性
│   ├── report    # Markdown 报告
│   └── all       # run + meta + report 一体化
├── eval_edu.py              # 主评估逻辑
├── compare_evals.py         # 对比工具
├── analyze_errors.py        # 错误归类
├── meta_evaluation.py       # 元评估
└── generate_report.py       # 报告生成

scripts/optimize/
├── edu_optimize.py          # 一站式入口（4 子命令）
│   ├── resample  # 数据重采样
│   ├── build     # GRPO 数据准备
│   ├── retrain   # 触发再训练
│   └── auto      # resample + retrain 一体化
├── resample_data.py         # v2 平滑公式
└── wandb_integration.py     # 训练监控
```

## 典型工作流

```bash
# 一站式评估
python scripts/eval/edu_evaluate.py all --stage sft --model_path out/edu_sft --eval_all

# 一站式优化
python scripts/optimize/edu_optimize.py auto --epochs 2 --save_weight edu_sft_v2
```

## 注意事项

1. 基线必须存在 — 退化检测依赖 `eval_results/baseline.json`
2. 评估集与训练集严格分离 — `download_all_data.py` 的 `create_eval_set()` 确保无重叠
3. vLLM 仅用于评估推理，训练使用 DDP/DeepSpeed/FSDP
4. 元评估每月至少 1 次 `meta_evaluation.py`