# QwenSearch — 亲子教育 VLM

基于 Qwen2-VL 的拍照做题 VLM，专为亲子教育场景设计。拍题即答，分步引导，亲子共学。

## 核心流程

```
下载模型 → 下载数据 → SFT 训练 → 评估 → 优化 → GRPO 训练 → 最终评估
```

## 快速开始

### 1. 环境

```bash
conda create -n qwensearch python=3.10 -y && conda activate qwensearch
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. 下载基座模型

```bash
pip install modelscope
modelscope download --model qwen/Qwen2-VL-2B-Instruct --local_dir ./model/Qwen2-VL-2B-Instruct
```

### 3. 下载数据集

```bash
python download_all_data.py       # 一键下载 22 训练 + 19 评估
```

### 4. 训练 + 评估 + 优化（一站式 5 步）

```bash
# ① SFT 训练（多卡自动 DDP）
python trainer/train_sft.py --epochs 3 --save_weight edu_sft

# ② SFT 评估（自动 vLLM 加速）
python scripts/eval/edu_evaluate.py all --stage sft --model_path out/edu_sft --eval_all

# ③ 优化（自动 DeepSpeed）
python scripts/optimize/edu_optimize.py auto --epochs 2 --save_weight edu_sft_v2

# ④ GRPO 训练（LLM-as-Judge 奖励）
python trainer/train_grpo.py --from_weight ../out/edu_sft --epochs 1 --api_model gpt-4o-mini

# ⑤ 最终评估
python scripts/eval/edu_evaluate.py all --stage full --model_path out/edu_grpo --eval_all
```

## 项目结构

```
qwensearch/
├── model/                     # Qwen2-VL 封装 + LoRA
├── dataset/                   # 训练/评估数据（运行 download_all_data.py 下载）
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
│   ├── download_all_data.py   # 一键下载
│   ├── convert_edu_data.py    # 数据集转换
│   └── web_demo.py            # Gradio Web Demo
├── EVAL_DESIGN.md             # 评估系统设计
├── DATA.md                    # 数据集详细说明
└── requirements.txt
```

## 训练管线

### SFT 阶段
22 个数据集（~222K 条），加权采样训练。多卡自动 DDP，支持 DeepSpeed/FSDP。

### GRPO 阶段
从 SFT 权重加载，使用 **LLM-as-Judge API** 作为奖励函数。模型生成 K 个候选回答，API 按 5 维度评分（准确性 0.40、完整性 0.20、引导性 0.20、流畅度 0.10、格式 0.10），组内标准化后计算策略损失。

```bash
# 使用 OpenAI API
python trainer/train_grpo.py --from_weight ../out/edu_sft --api_model gpt-4o-mini

# 使用兼容 API（如 DeepSeek）
python trainer/train_grpo.py --from_weight ../out/edu_sft --api_model deepseek-chat --api_base_url https://api.deepseek.com/v1 --api_key sk-xxx
```

## 评估系统

19 个评估数据集，支持 Bootstrap 置信区间 + p-value 显著性检验。详见 [EVAL_DESIGN.md](EVAL_DESIGN.md)。

## 数据集

22 个训练数据集（~222K），19 个评估数据集（~8,950）。中文图文占比 52.2%。详见 [DATA.md](DATA.md)。

## 硬件要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU | RTX 3090 (24GB) | A100 (40GB) |
| 显存 | 24 GB | 40 GB |
| 内存 | 32 GB | 64 GB |
| 硬盘 | 100 GB | 200 GB |

## wandb 监控

```bash
# 训练时启用
python trainer/train_sft.py --use_wandb --wandb_project QwenSearch
python trainer/train_grpo.py --use_wandb --wandb_project QwenSearch
```

## License

Apache 2.0 | 基座模型 [Qwen2-VL](https://github.com/QwenLM/Qwen-VL) 遵循其原始协议

## 致谢

- [MiniMind-V](https://github.com/jingyaogong/minimind-v) — 项目架构与 README 风格参考
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) — 基座多模态模型
- 22 个开源数据集作者（详见 [DATA.md](DATA.md)）