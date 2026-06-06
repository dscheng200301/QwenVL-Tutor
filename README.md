# QwenVL-Tutor — 亲子教育 VLM

基于 Qwen3-VL 的拍照做题 VLM，专为亲子教育场景设计。拍题即答，分步引导，亲子共学。

## 关键信息

| 项目 | 规格 |
|------|------|
| **Python** | 3.10+ |
| **PyTorch** | 2.5.1+ (CUDA 12.x) |
| **训练框架** | DeepSpeed ZeRO-2 / FSDP / DDP（自动选择） |
| **基座模型** | Qwen3-VL-2B-Instruct (~4GB) |
| **显卡要求** | 单卡 24GB+ 显存（A10G/A100/3090/4090 等） |
| **训练耗时** | SFT 约 2-4h/epoch，GRPO 约 1-2h/epoch（可中断续训） |
| **存储需求** | 数据集 ~50GB，模型+输出 ~20GB |

> **租显卡推荐**：国内推荐 [AutoDL](https://www.autodl.com/)、[恒源云](https://www.hyyq.com/)；海外 [Vast.ai](https://vast.ai/)、[Lambda Labs](https://lambdalabs.com/)。推荐配置：A10G 24G 或 RTX 4090 24G。

## 核心流程

```
下载模型 → 下载+转换数据 → 分离训练/评估集 → 基模评估 → SFT训练 → SFT评估 → SFT优化 → GRPO训练 → GRPO评估 → GRPO优化 → 最终评估
```

## 快速开始

### 1. 环境

```bash
conda create -n qwenvl-tutor python=3.10 -y && conda activate qwenvl-tutor
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. 下载基座模型

```bash
pip install modelscope
modelscope download --model qwen/Qwen3-VL-2B-Instruct --local_dir ./model/Qwen3-VL-2B-Instruct
```

### 3. 下载 + 转换 + 分离训练/评估集

```bash
# 一键完成：下载原始数据集 → 转换为 Parquet 格式 → 分离训练集/评估集
python download_all_data.py

# 仅下载训练部分（不创建评估集）
python download_all_data.py --train

# 仅下载指定数据集
python download_all_data.py --datasets scienceqa ceval

# 单独转换某个数据集（如手动添加新数据）
python scripts/convert_edu_data.py --dataset scienceqa --output dataset/edu_science.parquet
```

> `download_all_data.py` 内部调用 `scripts/convert_edu_data.py` 完成下载和格式转换，再调用 `create_eval_set()` 从训练集中**彻底分离**评估样本，确保训练/评估数据零重叠。输出目录：
>
> - 训练集：`dataset/edu_*.parquet`（22 个文件）
> - 评估集：`dataset/eval/eval_*.parquet`（19 个文件）

### 4. 基模评估 + 训练 + 优化（一站式 8 步）

```bash
# ① 基模评估（训练前基线，退化检测依赖此步）
python scripts/eval/edu_evaluate.py run --stage baseline --eval_all --max_samples 200

# ② SFT 训练（多卡自动 DDP，可选 --use_wandb 开启 wandb 监控）
python trainer/train_sft.py --epochs 3 --save_weight edu_sft --use_wandb --wandb_project QwenVL-Tutor

# ③ SFT 评估（自动 vLLM 加速）
python scripts/eval/edu_evaluate.py all --stage sft --model_path out/edu_sft --eval_all

# ④ SFT 优化（基于评估结果自动 resample + retrain）
python scripts/optimize/edu_optimize.py auto --epochs 2 --save_weight edu_sft_v2

# ⑤ GRPO 训练（LLM-as-Judge 奖励，可选 --use_wandb 开启 wandb 监控）
python trainer/train_grpo.py --from_weight ../out/edu_sft --epochs 1 --api_model gpt-4o-mini --use_wandb --wandb_project QwenVL-Tutor

# ⑥ GRPO 评估
python scripts/eval/edu_evaluate.py all --stage grpo --model_path out/edu_grpo --eval_all

# ⑦ GRPO 优化（基于评估结果自动决策：调整超参 / 回退 SFT 优化）
python scripts/optimize/edu_optimize.py grpo --eval_file eval_results/grpo_xxx.json

# ⑧ 最终评估
python scripts/eval/edu_evaluate.py all --stage full --model_path out/edu_grpo --eval_all
```

> **wandb 监控**：训练脚本内置 wandb/swanlab 支持，添加 `--use_wandb --wandb_project QwenVL-Tutor` 即可记录训练曲线。国内网络推荐使用 `swanlab`（`pip install swanlab`），参数相同。

## 项目结构

```
QwenVL-Tutor/
├── model/                     # Qwen3-VL 封装 + LoRA
├── dataset/                   # 训练/评估数据（Parquet 格式）
├── trainer/
│   ├── train_sft.py           # SFT 训练（DDP/DeepSpeed/FSDP）
│   ├── train_grpo.py          # GRPO 训练（LLM-as-Judge API 奖励）
│   ├── llm_reward.py          # LLM-as-Judge 奖励模型（API/vLLM/HF/混合）
│   ├── reward_model.py        # 规则化奖励模型（GRPO 优势函数 + 策略损失）
│   ├── trainer_utils.py       # 训练工具函数
│   ├── launch_distributed.py  # 分布式启动
│   └── terminal_dashboard.py  # 终端实时可视化
├── scripts/
│   ├── eval/
│   │   ├── edu_evaluate.py    # 一站式评估入口
│   │   ├── eval_edu.py        # 主评估（19 数据集 + 置信区间）
│   │   ├── compare_evals.py   # 对比 + p-value
│   │   ├── analyze_errors.py  # 错误归类
│   │   ├── meta_evaluation.py # 指标一致性
│   │   └── generate_report.py # Markdown 报告
│   ├── optimize/
│   │   ├── edu_optimize.py    # 一站式优化入口
│   │   ├── resample_data.py   # 数据重采样
│   │   └── wandb_integration.py
│   ├── convert_edu_data.py    # 数据集转换
│   └── web_demo.py            # Gradio Web Demo
├── download_all_data.py        # 一键下载+转换+分离训练/评估集
├── EVAL_DESIGN.md             # 评估系统设计
├── DATA.md                    # 数据集详细说明
└── requirements.txt
```

## 训练管线

### SFT 阶段

22 个数据集（\~222K 条），加权采样训练。多卡自动 DDP，支持 DeepSpeed/FSDP。

### GRPO 阶段

从 SFT 权重加载，使用 **LLM-as-Judge API** 作为奖励函数。模型生成 K 个候选回答，API 按 5 维度评分（准确性 0.40、完整性 0.20、引导性 0.20、流畅度 0.10、格式 0.10），组内标准化后计算策略损失。

```bash
# 使用 OpenAI API
python trainer/train_grpo.py --from_weight ../out/edu_sft --api_model gpt-4o-mini

# 使用兼容 API（如 DeepSeek）
python trainer/train_grpo.py --from_weight ../out/edu_sft --api_model deepseek-chat --api_base_url https://api.deepseek.com/v1 --api_key sk-xxx
```

## 评估系统

19 个评估数据集，支持 Bootstrap 置信区间 + p-value 显著性检验。详见 [EVAL\_DESIGN.md](EVAL_DESIGN.md)。

## 数据集

22 个训练数据集（\~222K），19 个评估数据集（\~8,950）。中文图文占比 52.2%。详见 [DATA.md](DATA.md)。

## 硬件要求

| 配置  | 最低              | 推荐          |
| --- | --------------- | ----------- |
| GPU | RTX 3090 (24GB) | A100 (40GB) |
| 显存  | 24 GB           | 40 GB       |
| 内存  | 32 GB           | 64 GB       |
| 硬盘  | 100 GB          | 200 GB      |

## License

Apache 2.0 | 基座模型 [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) 遵循其原始协议

***

> 更详细的技术细节（奖励模型设计、终端可视化、wandb 监控指标、数据分离机制等）见 **[README\_DETAILED.md](README_DETAILED.md)**。

## 致谢

- [MiniMind-V](https://github.com/jingyaogong/minimind-v) — 项目架构与 README 风格参考
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) — 基座多模态模型
- 22 个开源数据集作者（详见 [DATA.md](DATA.md)）

