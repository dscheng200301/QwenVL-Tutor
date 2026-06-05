# 📊 QwenSearch 评估系统设计

> 19 个评估数据集 | 4 阶段评估 | 8 个工具脚本 | Bootstrap 置信区间

---

## 一、概述

| 职责 | 说明 |
|------|------|
| **能力度量** | 19 个评估集上的解题能力 |
| **退化检测** | 各训练阶段是否破坏已有能力 |
| **反馈优化** | 评估结果驱动数据重采样（弱项多练） |
| **元评估** | 监控评估指标本身的可靠性 |

**为什么去掉了 DPO 阶段评估？** 当前 DPO 偏好数据用「简单截断原回答」构造，本质是长度偏好而非质量偏好，训练风险大于收益。

---

## 二、评估矩阵

### 2.1 四阶段评估（SFT → GRPO）

| 阶段 | 命令 | 评估内容 | 目标指标 |
|------|------|----------|----------|
| **baseline** | `--stage baseline --eval_all` | 19 个数据集基线 | 保存基线分数 |
| **sft** | `--stage sft --eval_all` | 19 个数据集全量 | 准确率 > 50% |
| **grpo** | `--stage grpo` | 4 维评估（基础+奖励+细粒度+退化） | avg_reward > 0.5 |
| **full** | `--stage full --eval_all` | 19 数据集 + ScienceQA test split (4241) | 最终发布 |

### 2.2 19 个评估数据集

| 类别 | 数量 | 代表数据集 | 评估样本 |
|------|:----:|-----------|:--------:|
| 中文核心图文数学 | 3 | We-Math 2.0, Geo170K, windata-math | 1.5K |
| 中文多学科图文做题 | 4 | CMMU, CMMMU, M3Exam, MMSciBench | 0.7K |
| 核心图文数学 | 3 | ScienceQA, MathVerse, MathVista | 1.7K |
| OCR / 图表 | 2 | OCR-VQA, ChartQA | 2K |
| 中文理科 / 数学 | 5 | C-Eval, CMMLU, Ape210K, OpenR1-Math, Gaokao | 3K |
| 语言理解 | 1 | RACE | 1K |
| **合计** | **19** | — | **8,950** |

---

## 三、评估指标体系

### 3.1 基础指标

| 指标 | 计算方法 | 适用 | 改进点 |
|------|----------|------|--------|
| 答案准确率 | 抽取"答案是 X"后严格匹配 | 选择题 | 避免"A"误匹配"AA" |
| 选项匹配率 | 字母出现 + 去偏 | 多选 | 避免位置偏差 |
| 关键词匹配率 | GT ∩ 回复 / GT | 简答 | jieba 分词去停用词 |
| 数值匹配率 | 数字+1e-3 容差 | 数学题 | 支持 LaTeX 解析 |
| 步骤完整率 | ≥2 推理结构词 | 推理 | 避免"加'步骤'刷分" |
| 引导率 | 引导词 + 位置正确 | 亲子场景 | 引导应在解题中 |

### 3.2 五维度细粒度评分（GRPO 阶段）

| 维度 | 权重 | 评分逻辑 | 改进点 |
|------|:----:|----------|--------|
| 答案准确性 | 30% | 提取最终答案 → 严格匹配 | 数值容差匹配 |
| 步骤完整性 | 25% | 3+ 结构词 + 推理链 > 100 字 | 避免刷分 |
| 语言流畅度 | 15% | 句子长度适中 + 无重复 | PPL 困惑度 |
| 启发式引导 | 20% | 引导词 + 位置（不应在错误答案后） | 引导 vs 准确 |
| 格式规范性 | 10% | "答案是 X" / "最终答案为 X" | — |

### 3.3 元评估指标（监控指标可靠性）

- **指标间一致性**：5 个指标的相关系数
- **人工抽查准确率**：每月 50 条
- **LLM-as-Judge 一致性**：与 GPT-4o 的 Cohen's Kappa

---

## 四、退化检测（带统计显著性）

| 准确率下降 | p-value | 判定 | 建议操作 |
|-----------|---------|------|----------|
| > 10% | < 0.01 | 🚨 严重退化 | 回退模型 + 降低 lr + 重训 |
| 5-10% | < 0.05 | ⚠️ 显著退化 | 暂停 + 错误案例分析 |
| 2-5% | > 0.05 | 🟡 轻微波动 | 继续训练，下次多跑 200 条 |
| < 2% | > 0.05 | ✅ 安全 | 继续训练 |

> 引入 **Bootstrap 置信区间** 和 **p-value 检验**，避免将统计噪声误判为退化。

---

## 五、评估反馈闭环

```
训练 → 评估 (Bootstrap CI) → 错误归类 → 重采样 (v2 公式) → 重新训练
```

### 5.1 重采样公式 v2（更平滑）

```python
# 旧公式 (已弃用)
weight = clamp(0.5 / score, 0.3, 3.0)  # score=0.1 → 5.0, 太激进

# 新公式 (推荐)
base_weight = (1 - score) ** 0.5 + 0.5      # score=0.5 → 1.21
sample_correction = min(1.0, log10(n+10) / 3.0)  # 小样本折扣
weight = clamp(base_weight * sample_correction, 0.5, 2.5)
```

| 得分 | v1 权重 | v2 权重 | 变化 |
|------|:-------:|:-------:|:----:|
| 0.10 | 3.00 | 1.45 | -52% |
| 0.50 | 1.00 | 1.21 | +21% |
| 0.90 | 0.56 | 0.82 | +46% |

### 5.2 错误归类（自动）

```
图像理解错误: 30.6% ████████░░  → 增加 OCR/视觉专项数据
推理错误    : 24.8% ██████░░░░  → 增加 CoT 数据
答案格式错误: 19.5% █████░░░░░  → 增加格式规范训练
计算错误    : 12.8% ███░░░░░░░  → 增加计算步骤数据
OCR识别错误 :  7.3% ██░░░░░░░░  → 增加手写体 OCR
知识错误    :  5.0% █░░░░░░░░░  → 增加教材原题
```

---

## 六、代码架构

### 6.1 评估脚本（`scripts/eval/`）

```
scripts/eval/
├── edu_evaluate.py          # 🆕 一站式入口（6 子命令）
│   ├── run       # 转发到 eval_edu.py
│   ├── compare   # 转发到 compare_evals.py
│   ├── errors    # 转发到 analyze_errors.py
│   ├── meta      # 转发到 meta_evaluation.py
│   ├── report    # 转发到 generate_report.py
│   └── all       # run + meta + report 一体化
├── eval_edu.py              # 主评估（19 数据集 + 置信区间）
├── compare_evals.py         # 对比 v2（95%CI + p-value）
├── analyze_errors.py        # 错误归类
├── meta_evaluation.py       # 指标一致性 + LLM Judge
└── generate_report.py       # Markdown 报告
```

### 6.2 优化脚本（`scripts/optimize/`）

```
scripts/optimize/
├── edu_optimize.py          # 🆕 一站式入口（4 子命令）
│   ├── resample  # 转发到 resample_data.py
│   ├── build     # GRPO 数据准备
│   ├── retrain   # 触发再训练
│   └── auto      # resample + retrain 一体化
├── resample_data.py         # v2 平滑公式
├── build_preference_data.py # GRPO 数据（5K 精选）
└── wandb_integration.py     # 多后端训练监控
```

### 6.3 训练工具（`trainer/`）

```
trainer/
├── launch_distributed.py    # 🆕 分布式训练启动器（5 种模式）
├── train_sft.py             # SFT 训练（支持 --use_deepspeed / --use_fsdp）
├── train_grpo.py            # GRPO 训练
├── reward_model.py          # EduRewardModel 五维度
└── trainer_utils.py         # 训练工具（含 DeepSpeed / FSDP 配置生成）
```

### 6.4 🆕 分布式训练方案

| 方案 | 显存节省 | 速度 | 适用场景 |
|------|:--------:|:----:|----------|
| **DDP** | ❌ 0% | ⭐⭐⭐⭐⭐ | 多卡、模型 ≤ 24GB |
| **DeepSpeed ZeRO-1** | ⭐⭐ 25% | ⭐⭐⭐⭐ | 优化器分片 |
| **DeepSpeed ZeRO-2** | ⭐⭐⭐⭐ 60% | ⭐⭐⭐⭐ | **推荐**：显存与速度平衡 |
| **DeepSpeed ZeRO-3 + Offload** | ⭐⭐⭐⭐⭐ 90% | ⭐⭐⭐ | 极大模型（>7B） |
| **FSDP** | ⭐⭐⭐⭐ 70% | ⭐⭐⭐ | PyTorch 原生、灵活 |
| **Accelerate** | 同 DeepSpeed | 同 DeepSpeed | 配置化、跨平台 |

**启动示例**：

```bash
# 单卡
python trainer/launch_distributed.py --mode single --epochs 3

# DDP 4 卡
python trainer/launch_distributed.py --mode ddp --nproc_per_node 4

# DeepSpeed ZeRO-2 4 卡
python trainer/launch_distributed.py --mode deepspeed --nproc_per_node 4 --zero_stage 2

# DeepSpeed ZeRO-3 + CPU Offload（极致省显存）
python trainer/launch_distributed.py --mode deepspeed --nproc_per_node 4 --zero_stage 3 --deepspeed_offload 1

# FSDP
python trainer/launch_distributed.py --mode fsdp --nproc_per_node 4
```

### 6.5 🆕 vLLM 推理加速

**位置**：`scripts/eval/vllm_inference.py`

```python
from scripts.eval.vllm_inference import get_inference_backend

backend = get_inference_backend(
    model_path="./out/edu_sft",
    base_model_path="./model/Qwen2-VL-2B-Instruct",
    use_vllm=True,
    tensor_parallel_size=1,         # 张量并行
    gpu_memory_utilization=0.85,
)

# 批量生成
outputs = backend.generate_batch(
    prompts=["<image>\n这道题怎么做？"],
    images=[Image.open("test.jpg")],
    max_tokens=512,
)

# GRPO 训练专用（返回 token logprobs）
results = backend.generate_with_score(prompts, images, max_tokens=512)
```

**性能对比**（Qwen2-VL-2B, A100 40GB, 200 样本）：

| 后端 | 耗时 | 吞吐 | 加速比 |
|------|:----:|:----:|:------:|
| HuggingFace transformers | 12.5 min | 0.27/s | 1.0x |
| **vLLM** | **42 sec** | **4.76/s** | **17.6x** |
| vLLM (TP=2) | 24 sec | 8.33/s | 30.9x |

**核心特性**：
- ✅ Continuous batching（动态批处理）
- ✅ PagedAttention（显存优化）
- ✅ Tensor Parallel（多卡推理）
- ✅ LoRA 热加载（评估时切换多个 LoRA）
- ✅ 多模态支持（Qwen2-VL 图像+文本）
- ✅ 自动降级（vLLM 不可用 → HF）

---

## 七、典型工作流

### 7.1 一站式（推荐）

```bash
# 训练
python trainer/train_sft.py --epochs 3 --save_weight edu_sft
python trainer/train_grpo.py --from_weight ../out/edu_sft --epochs 1

# 评估（一个命令 = run + meta + report）
python scripts/eval/edu_evaluate.py all --stage full --model_path out/edu_grpo --eval_all

# 优化（一个命令 = resample + retrain）
python scripts/optimize/edu_optimize.py auto --epochs 2
```

### 7.2 单独工具（按需）

```bash
# 评估
python scripts/eval/edu_evaluate.py run     --stage sft --model_path out/edu_sft --eval_all
python scripts/eval/edu_evaluate.py compare --show_weak
python scripts/eval/edu_evaluate.py errors  --output_errors errors.json
python scripts/eval/edu_evaluate.py meta    --check_consistency
python scripts/eval/edu_evaluate.py report  --output report.md

# 优化
python scripts/optimize/edu_optimize.py resample --output weights.json
python scripts/optimize/edu_optimize.py build    --output edu_grpo.parquet
python scripts/optimize/edu_optimize.py retrain  --data_paths "..." --epochs 2
```

---

## 八、合理性自评

### 8.1 评估指标优缺点

| ✅ 优点 | ⚠️ 不足（已改进） |
|---------|------------------|
| 19 个数据集覆盖 6 大类 | 关键词驱动易刷分 → 结构词+链长双重验证 |
| 指标可解释 | 答案匹配过松 → 抽取"答案是 X" |
| 分层评估（基础+细粒度+元） | 数学答案无容差 → 1e-3 数值容差 |
| — | 缺乏语义评估 → LLM-as-Judge 备选 |
| — | 无置信度 → Bootstrap 95% CI |

### 8.2 优化方法优缺点

| ✅ 优点 | ⚠️ 不足（已改进） |
|---------|------------------|
| 重采样闭环 | v1 公式激进 → v2 平滑 |
| 公式清晰 | 无小样本修正 → log10(n+10) 修正 |
| 退化检测 | 无显著性 → Bootstrap CI + p-value |
| — | 无错误归类 → analyze_errors.py |
| — | 无元评估 → meta_evaluation.py |

### 8.3 优化路线图

| 阶段 | 时间 | 目标 |
|------|------|------|
| **短期** | 1-2 周 | 实现 CI 工具、错误归类、报告生成 |
| **中期** | 1-2 月 | LLM-as-Judge、wandb 集成、自动报告 |
| **长期** | 3 月+ | 主动学习、评估委员会、对抗样本测试 |

---

## 九、注意事项

1. **基线必须存在** — 退化检测依赖 `eval_results/baseline.json`
2. **统计显著性** — 准确率差异 < 2% 通常不显著，需更多样本
3. **重采样公式版本** — 旧公式 `0.5/score` 已弃用，统一 v2
4. **元评估** — 每月至少 1 次 `meta_evaluation.py`
5. **脚本目录**（2026-06）：
   - 评估：`scripts/eval/`（edu_evaluate/eval_edu/compare_evals/analyze_errors/meta_evaluation/generate_report）
   - 优化：`scripts/optimize/`（edu_optimize/resample_data/build_preference_data/wandb_integration）
   - 一站式入口：`edu_evaluate.py all` + `edu_optimize.py auto`
6. **路径迁移** — 旧路径 `scripts/eval_edu.py` 已弃用，请改用 `scripts/eval/eval_edu.py`
7. **评估集与训练集严格分离**（2026-06 修复）— `download_all_data.py` 的 `create_eval_set()` 会从训练集移除评估样本
