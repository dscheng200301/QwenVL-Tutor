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

```
Python 3.10+ · PyTorch 2.4+ · transformers>=4.45 · peft>=0.10
accelerate>=0.30 · datasets>=2.20 · pyarrow>=15 · Pillow>=10
```

### ⏱️ 预估时间（完整流程）

| GPU | SFT (3 ep) | GRPO (1 ep) | 评估 (19×200) | 总计 |
|-----|:----------:|:-----------:|:-------------:|:----:|
| RTX 3090 | 36-48 h | 6-10 h | 3-5 h | **50-70 h** |
| A100 40GB | 12-18 h | 2-4 h | 1-2 h | **18-28 h** |
| A100 80GB | 8-12 h | 1-2 h | 1 h | **12-18 h** |

> 配置：`batch_size=4 × grad_accum=8`（有效 batch=32），`max_seq_len=2048`

### ⚡ 分布式训练 + vLLM 推理

> **🚀 性能提升**：多卡 DDP/DeepSpeed/FSDP + vLLM 推理加速 5-20x

#### 🆕 零配置自动加速（一站式脚本默认启用）

```bash
# 训练：自动检测 GPU 数量 → DeepSpeed ZeRO-2/3
python scripts/optimize/edu_optimize.py auto --epochs 2 --save_weight edu_sft_v2
# 输出:
#   自动加速决策
#   ===============
#     检测到 4 张 GPU（最小显存 40.0 GB）
#     -> 分布式训练模式: deepspeed (ZeRO-2)
#     -> vLLM 推理: 已启用（TP=2）

# 评估：自动检测 vLLM 可用性 → TP 自动推荐
python scripts/eval/edu_evaluate.py all --stage sft --model_path out/edu_sft --eval_all
# 输出:
#   vLLM 推理决策
#   ===============
#     检测到 4 张 GPU
#     vLLM 安装: 是
#     -> vLLM 推理: 已启用（TP=2）

# 手动覆盖
python scripts/optimize/edu_optimize.py retrain --epochs 2 --no_distributed
python scripts/eval/edu_evaluate.py run --stage sft --model_path out/edu_sft --no_vllm
```

#### 手动分布式训练启动器

```bash
# === 单卡 ===
python trainer/launch_distributed.py --mode single --epochs 3

# === DDP 数据并行（多卡）===
python trainer/launch_distributed.py --mode ddp --nproc_per_node 4

# === DeepSpeed ZeRO-2（推荐，节省显存）===
python trainer/launch_distributed.py --mode deepspeed --nproc_per_node 4 --zero_stage 2

# === DeepSpeed ZeRO-3（大模型必备）===
python trainer/launch_distributed.py --mode deepspeed --nproc_per_node 4 --zero_stage 3

# === DeepSpeed + CPU Offload（极致省显存）===
python trainer/launch_distributed.py --mode deepspeed --nproc_per_node 4 --zero_stage 3 --deepspeed_offload 1

# === FSDP 全分片（PyTorch 内置）===
python trainer/launch_distributed.py --mode fsdp --nproc_per_node 4

# === Accelerate Launch（通用）===
python trainer/launch_distributed.py --mode accelerate --nproc_per_node 4
```

#### vLLM 推理加速（评估时）

```python
from scripts.eval.vllm_inference import get_inference_backend

# 自动选择 vLLM（不可用时降级到 HF）
backend = get_inference_backend(
    model_path="./out/edu_sft",
    base_model_path="./model/Qwen2-VL-2B-Instruct",
    use_vllm=True,
    tensor_parallel_size=1,         # 张量并行 GPU 数
    gpu_memory_utilization=0.85,    # 显存使用率
)

# 批量推理（带图像）
from PIL import Image
outputs = backend.generate_batch(
    prompts=["<image>\n这道题怎么做？"],
    images=[Image.open("test.jpg")],
    max_tokens=512,
    temperature=0.0,
)
print(outputs[0])
```

**性能对比**（Qwen2-VL-2B, 单卡 A100 40GB, 200 样本）：

| 后端 | 耗时 | 吞吐 |
|------|:----:|:----:|
| HuggingFace transformers | 12.5 min | 0.27 样本/秒 |
| **vLLM** | **42 sec** | **4.76 样本/秒** (↑17.6x) |

详细配置见 [EVAL_DESIGN.md](EVAL_DESIGN.md#六代码架构)

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
│   │   ├── build_preference_data.py
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

> **🆕 零配置自动加速**：`edu_optimize.py` 自动检测 GPU 数量启用 DeepSpeed，`edu_evaluate.py` 自动启用 vLLM

```bash
# ① SFT 训练
python trainer/train_sft.py --epochs 3 --save_weight edu_sft

# ② 一站式 SFT 评估（🆕 自动 vLLM 加速）
python scripts/eval/edu_evaluate.py all --stage sft --model_path out/edu_sft --eval_all

# ③ 一站式优化（🆕 自动检测 4 卡 → DeepSpeed ZeRO-2）
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
