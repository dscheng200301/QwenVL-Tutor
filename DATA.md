# 📚 QwenSearch 数据集来源说明

> 本文档详细记录 QwenSearch 项目使用的数据集来源、许可协议、下载方式和使用说明。

---

## � 快速开始

### 一键下载所有数据集

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
├── edu_*.parquet              # 训练数据集
│   ├── edu_science.parquet     # ScienceQA (6,218 条)
│   ├── edu_ceval.parquet       # C-Eval (2,654 条)
│   ├── edu_windata_math.parquet # windata-math (10,000 条)
│   └── ...
│
├── eval/                      # 评估数据集
│   ├── eval_science.parquet    # ScienceQA 评估集 (500 条)
│   ├── eval_ceval.parquet      # C-Eval 评估集 (200 条)
│   └── ...
│
├── edu_dataset.py              # 数据集加载器
└── __init__.py                # 模块初始化
```

---

## 📋 数据集总览

| 类别 | 数据集 | 语言 | 训练条数 | 评估条数 | 开源协议 |
|------|--------|:----:|:--------:|:--------:|----------|
| **核心中文图文数学（新增）** | We-Math 2.0 | CN | ~10,000 | 500 | CC BY 4.0 |
| | Geo170K | CN | 50,000 | 500 | Apache 2.0 |
| | CMM-Math | CN | 28,069 | 500 | CC BY 4.0 |
| | MathReal | CN | 2,000 | 200 | MIT |
| **核心图文数学** | ScienceQA | EN | 6,218 | 500 | CC BY 4.0 |
| | MathVerse | EN | 3,940 | 200 | CC BY 4.0 |
| | MathVista | EN | 1,000 | 100 | MIT |
| **中文图文** | windata-math | CN | 10,000 | 500 | CC-BY-4.0 |
| **OCR 识别** | OCR-VQA | EN | 20,000 | 500 | Apache 2.0 |
| **图表理解** | ChartQA | EN | 10,000 | 500 | MIT |
| **中文理科** | C-Eval | CN | 2,654 | 200 | Apache 2.0 |
| | CMMLU | CN | 11,917 | 500 | CC BY-NC 4.0 |
| **中文数学** | Ape210K | CN | 20,000 | 500 | MIT |
| | OpenR1-Math K12 | CN | 20,000 | 500 | Apache 2.0 |
| | Gaokao MathQA | CN | 351 | 100 | CC BY 4.0 |
| | Gaokao MathCloze | CN | 118 | 50 | CC BY 4.0 |
| **语言理解** | RACE | CN/EN | 10,000 | 500 | MIT |
| | **合计** | | **~206,267** | **4,650** | |

---

## 🆕 核心中文图文数学数据集（新增）

### 1. We-Math 2.0

**系统性数学知识体系数据集**，专为多模态数学推理设计。

| 属性 | 说明 |
|------|------|
| **数据集名称** | We-Math 2.0: A Versatile MathBook System |
| **语言** | 中文 |
| **数据规模** | ~10,000 条 |
| **数据类型** | 图文数学问答 |
| **学科覆盖** | 小学到高中，491 个知识点，1819 个知识原理 |
| **来源平台** | HuggingFace |
| **仓库地址** | [We-Math/We-Math2.0-Standard](https://huggingface.co/datasets/We-Math/We-Math2.0-Standard) |
| **许可协议** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **论文引用** | [We-Math 2.0](https://arxiv.org/abs/2508.10433) |
| **发布时间** | 2025年 |
| **维护团队** | 多所高校联合发布 |

**数据特点**：
- ✅ 系统性知识体系：5 个层级、491 个知识点、1819 个知识原理
- ✅ 双向数据扩展："一题多图" 和 "一图多题"
- ✅ 覆盖小学到高中，包含部分大学及竞赛知识
- ✅ 手动利用 Geogebra 专业化软件渲染高质量图像
- ✅ 每道题目都标注了对应的多层级知识点

**下载方式**：
```bash
# 使用转换脚本
python scripts/convert_edu_data.py --dataset we_math
```

---

### 2. Geo170K

**几何推理专项数据集**，华为诺亚方舟实验室发布。

| 属性 | 说明 |
|------|------|
| **数据集名称** | Geo170K: Geometric Math Dataset |
| **语言** | 中文 |
| **数据规模** | 170,000+ 条（计划采样 50,000 条） |
| **数据类型** | 几何图文问答 |
| **学科覆盖** | 平面几何、立体几何、解析几何 |
| **来源平台** | HuggingFace |
| **仓库地址** | [Luckyjhg/Geo170K](https://huggingface.co/datasets/Luckyjhg/Geo170K) |
| **许可协议** | Apache License 2.0 |
| **论文引用** | [G-LLaVA](https://arxiv.org/abs/2312.11370) |
| **发布时间** | 2023年 |
| **维护团队** | 华为诺亚方舟实验室、香港大学、港科大 |

**数据特点**：
- ✅ 超大规模几何数据集：约 6 万张几何图像，11 万+ 问答对
- ✅ 涵盖基本几何要素判定、定量关系、符号推理
- ✅ 数据来源于 Geometry3K、GeoQA、GeoQA+ 等多个数据集
- ✅ 非常适合训练几何空间推理能力
- ✅ 开源可商用 (Apache 2.0)

**下载方式**：
```bash
# 使用转换脚本（采样 50,000 条）
python scripts/convert_edu_data.py --dataset geo170k
```

---

### 3. CMM-Math

**中文 K12 数学图文数据集**，华东师范大学发布。

| 属性 | 说明 |
|------|------|
| **数据集名称** | CMM-Math: Chinese Multimodal Math |
| **语言** | 中文 |
| **数据规模** | 28,069 条（含 6,869 条多模态） |
| **数据类型** | 图文数学问答 |
| **学科覆盖** | 小学到高中 12 个年级 |
| **来源平台** | GitHub |
| **仓库地址** | [ECNU-ICALK/EduChat-Math](https://github.com/ECNU-ICALK/EduChat-Math) |
| **许可协议** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **论文引用** | [CMM-Math](https://arxiv.org/abs/2409.02834) |
| **发布时间** | 2024年 |
| **维护团队** | 华东师范大学 |

**数据特点**：
- ✅ 专为中文 K12 数学设计
- ✅ 包含选择题、填空题、判断题、分析题
- ✅ 约 6,869 条含图片（多模态）
- ✅ 包含详细的解题步骤
- ✅ 学术数据集，质量高
- ✅ 开源可商用 (CC BY 4.0)

**下载方式**：
```bash
# 使用转换脚本
python scripts/convert_edu_data.py --dataset cmm_math
```

---

### 4. MathReal

**真实场景 K12 数学图文数据集**。

| 属性 | 说明 |
|------|------|
| **数据集名称** | MathReal: Real Scene Math |
| **语言** | 中文 |
| **数据规模** | 2,000 条 |
| **数据类型** | 真实场景拍摄数学图文 |
| **图像特点** | 手持设备拍摄，含多种退化 |
| **来源平台** | GitHub |
| **仓库地址** | [junfeng0288/MathReal](https://github.com/junfeng0288/MathReal) |
| **许可协议** | MIT License |
| **论文引用** | [MathReal](https://arxiv.org/abs/2508.06009) |
| **发布时间** | 2025年 |

**数据特点**：
- ✅ 真实场景拍摄：通过手持移动设备采集
- ✅ 多样化退化：图像质量损失、视角变化、无关内容干扰
- ✅ 覆盖 3 种退化类型、14 个子类别
- ✅ 涵盖 5 个知识领域、3 种题型、3 级难度
- ✅ 非常适合训练真实场景理解能力
- ✅ 开源可商用 (MIT)

**下载方式**：
```bash
# 使用转换脚本
python scripts/convert_edu_data.py --dataset math_real
```

---

## 🔬 核心图文数学数据集

### 1. ScienceQA

**核心多模态科学问答数据集**，包含丰富的图文内容和详细的 lecture 解析。

| 属性 | 说明 |
|------|------|
| **数据集名称** | Science Question Answering Benchmark |
| **语言** | 英文 |
| **数据规模** | 6,218 条 |
| **数据类型** | 图文问答 + lecture 解析 |
| **学科覆盖** | 自然科学（物理、化学、生物）、社会科学、艺术 |
| **来源平台** | HuggingFace |
| **仓库地址** | [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) |
| **许可协议** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **论文引用** | [ScienceQA: Multimodal Science Question Answering](https://arxiv.org/abs/2210.11431) |
| **发布时间** | 2022年 |
| **维护团队** | UCLA, Allen Institute for AI |

**数据特点**：
- 每个问题包含图片、题目、选项和详细解释
- 涵盖 3 个学科领域，26 个主题
- 适合训练模型的科学推理能力

**下载方式**：
```bash
# 方式 1: 使用转换脚本
python scripts/convert_edu_data.py --dataset scienceqa

# 方式 2: 直接从 HuggingFace 下载
from datasets import load_dataset
dataset = load_dataset("derek-thomas/ScienceQA", split="test")
```

**使用说明**：
- 用于 SFT 阶段的全阶段训练
- 用于 DPO 阶段的偏好数据生成
- 用于 GRPO 阶段的奖励信号计算
- 建议采样比例：100%（全量使用）

---

### 2. MathVerse

**数学视觉推理专业数据集**，专注于数学问题的多模态理解。

| 属性 | 说明 |
|------|------|
| **数据集名称** | MathVerse: Math Video Understanding Benchmark |
| **语言** | 英文 |
| **数据规模** | 3,940 条 |
| **数据类型** | 数学图表 + 问答 |
| **数学领域** | 算术、代数、几何、概率 |
| **来源平台** | HuggingFace |
| **仓库地址** | [AI4Math/MathVerse](https://huggingface.co/datasets/AI4Math/MathVerse) |
| **许可协议** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **论文引用** | [MathVista: Mathematical Visual Reasoning Benchmark](https://arxiv.org/abs/2308.06358) |
| **发布时间** | 2024年 |
| **维护团队** | UCLA, Microsoft Research |

**数据特点**：
- 专注于数学图表的理解和推理
- 包含多种图表类型（几何图形、函数图像、统计图等）
- 适合提升模型的数学视觉能力

**下载方式**：
```bash
python scripts/convert_edu_data.py --dataset math_verse
```

**使用说明**：
- 仅用于 SFT 阶段训练
- 建议采样比例：100%（全量使用）

---

### 3. MathVista

**数学视觉推理基准测试集**，评估模型的数学视觉理解能力。

| 属性 | 说明 |
|------|------|
| **数据集名称** | MathVista: Mathematical Visual Reasoning Benchmark |
| **语言** | 英文 |
| **数据规模** | 1,000 条 |
| **数据类型** | 数学图表 + 问答 |
| **任务类型** | 选择题、数值题、证明题 |
| **来源平台** | HuggingFace |
| **仓库地址** | [AI4Math/MathVista](https://huggingface.co/datasets/AI4Math/MathVista) |
| **许可协议** | MIT License |
| **论文引用** | [MathVista: Mathematical Visual Reasoning Benchmark](https://arxiv.org/abs/2308.06358) |
| **发布时间** | 2023年 |
| **维护团队** | UCLA, Microsoft Research |

**数据特点**：
- 多样化的数学视觉问题
- 包含真实的数学教材截图
- 涵盖 K-12 到大学水平的数学问题

**下载方式**：
```bash
python scripts/convert_edu_data.py --dataset math_vista
```

**使用说明**：
- 仅用于 SFT 阶段训练
- 建议采样比例：100%（全量使用）

---

## 🇨🇳 中文图文数据集

### 4. windata-math

**中文数学图文推理数据集**，来自 windata-vision-synthetics-zh-300k 的 geo170k 子集。

| 属性 | 说明 |
|------|------|
| **数据集名称** | windata-vision-synthetics-zh-300k (geo170k subset) |
| **语言** | 中文 |
| **数据规模** | 10,000 条 |
| **数据类型** | 数学图表 + 问答 |
| **图像数量** | ~10,000 张 |
| **来源平台** | ModelScope |
| **仓库地址** | [wair/windata-vision-synthetics-zh-300k](https://modelscope.cn/datasets/wair/windata-vision-synthetics-zh-300k) |
| **许可协议** | Creative Commons Attribution 4.0 International (CC-BY-4.0) |
| **发布机构** | 卫宁健康人工智能实验室 (WINNING AI Lab) |
| **发布时间** | 2024年12月 |

**数据来源构成**：

| 子数据集 | 数据量 | 说明 |
|----------|--------|------|
| Geo170KQA | 75,000 | 几何图形推理问答（已采样 10,000） |
| PlotQA | 38,437 | 图表数据问答 |
| TallyQA | 81,950 | 文档问答 |
| Docmatix | 61,817 | 文档 caption/OCR |

**数据特点**：
- 专为中文多模态场景设计
- 包含丰富的几何图形和数学图表
- 对话格式，天然适配 SFT 训练
- 经过规则过滤，质量高

**下载和转换方式**：
```bash
# 1. 下载数据集
git clone https://modelscope.cn/datasets/wair/windata-vision-synthetics-zh-300k dataset/windata-300k

# 2. 解压图像数据
cd dataset/windata-300k
tar -xzf geo170k.tar.gz

# 3. 转换格式
python scripts/convert_windata.py \
    --input dataset/windata-300k \
    --output dataset/edu_windata_math.parquet \
    --max_samples 10000
```

**使用说明**：
- 仅用于 SFT 阶段训练
- 建议采样比例：100%（已限制为 10,000 条）
- 注意：需要解压 geo170k.tar.gz 后再转换

---

## 🔤 OCR 和图表理解数据集

### 5. OCR-VQA

**OCR 视觉问答数据集**，训练模型从图像中读取和理解文本的能力。

| 属性 | 说明 |
|------|------|
| **数据集名称** | OCR-VQA: Visual Question Answering on Printed and Handwritten Text |
| **语言** | 英文 |
| **数据规模** | 20,000 条 |
| **数据类型** | 书籍封面/海报 + OCR 问答 |
| **图像来源** | 书籍封面、海报、CD 封面等 |
| **来源平台** | HuggingFace |
| **仓库地址** | [MMInstruction/OCR-VQA](https://huggingface.co/datasets/MMInstruction/OCR-VQA) |
| **许可协议** | Apache License 2.0 |
| **论文引用** | [OCR-VQA: Visual Question Answering on Printed Text](https://arxiv.org/abs/1909.10720) |
| **发布时间** | 2019年 |

**数据特点**：
- 图像中包含大量印刷文本
- 问题需要从图像中读取文本并理解
- 适合训练 OCR 和文本理解能力

**下载方式**：
```bash
python scripts/convert_edu_data.py --dataset ocr
```

**使用说明**：
- 仅用于 SFT 阶段训练
- ⚠️ 建议降权 20-30%（与核心数学任务关联度较低）
- 可用于提升模型的文本读取能力

---

### 6. ChartQA

**图表问答数据集**，包含柱状图、折线图、饼图等多种图表。

| 属性 | 说明 |
|------|------|
| **数据集名称** | Chart Question Answering |
| **语言** | 英文 |
| **数据规模** | 10,000 条 |
| **数据类型** | 数据图表 + 问答 |
| **图表类型** | 柱状图、折线图、饼图、散点图 |
| **来源平台** | HuggingFace |
| **仓库地址** | [HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) |
| **许可协议** | MIT License |
| **论文引用** | [ChartQA: A Benchmark for Question Answering about Charts](https://arxiv.org/abs/2203.10216) |
| **发布时间** | 2022年 |
| **维护团队** | Microsoft Research |

**数据特点**：
- 图表数据丰富，包含真实和合成图表
- 问题涉及数据提取、计算、推理
- 适合训练数据理解和推理能力

**下载方式**：
```bash
python scripts/convert_edu_data.py --dataset chartqa
```

**使用说明**：
- 仅用于 SFT 阶段训练
- 建议采样比例：100%（全量使用）

---

## 🇨🇳 中文理科数据集

### 7. C-Eval

**中文综合学科评估数据集**，涵盖 14 个学科领域。

| 属性 | 说明 |
|------|------|
| **数据集名称** | C-Eval: A Multi-level Multi-discipline Chinese Evaluation Benchmark |
| **语言** | 中文 |
| **数据规模** | 2,654 条 |
| **数据类型** | 纯文本选择题 |
| **学科数量** | 14 个学科 |
| **来源平台** | HuggingFace |
| **仓库地址** | [ceval/ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam) |
| **许可协议** | Apache License 2.0 |
| **论文引用** | [C-Eval: A Multi-level Multi-discipline Chinese Evaluation Benchmark](https://arxiv.org/abs/2305.08322) |
| **发布时间** | 2023年 |
| **维护团队** | 复旦大学 |

**学科覆盖**：
```
基础科学: 初中数学、高中数学、大学数学、大学物理、大学化学
人文学科: 马克思主义基本原理、中华民族共同体概论
社会科学: 法学、教育学、经济学、会计学
医学: 临床医学、护士资格
```

**下载方式**：
```bash
python scripts/convert_edu_data.py --dataset ceval
```

**使用说明**：
- 仅用于 SFT 阶段训练
- 建议采样比例：100%（全量使用）
- 用于评估模型的中文理科能力

---

### 8. CMMLU

**中文多学科多语言评估数据集**，涵盖 67 个学科领域。

| 属性 | 说明 |
|------|------|
| **数据集名称** | CMMLU: A Massive Chinese Multi-modal Language Understanding Evaluation Benchmark |
| **语言** | 中文 |
| **数据规模** | 11,917 条 |
| **数据类型** | 纯文本选择题 |
| **学科数量** | 67 个学科 |
| **来源平台** | HuggingFace |
| **仓库地址** | [haonan-li/CMMLU](https://huggingface.co/datasets/haonan-li/CMMLU) |
| **许可协议** | Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) |
| **论文引用** | [CMMLU: A Massive Chinese Multi-modal Language Understanding Benchmark](https://arxiv.org/abs/2304.12547) |
| **发布时间** | 2023年 |

**学科分类**：

| 类别 | 学科数量 | 示例 |
|------|----------|------|
| STEM | ~20 | 数学、物理、化学、计算机科学 |
| 社会科学 | ~15 | 经济、心理、教育、法律 |
| 人文 | ~15 | 历史、哲学、文学、艺术 |
| 其他 | ~17 | 医学、农业、工程等 |

**下载方式**：
```bash
python scripts/convert_edu_data.py --dataset cmmlu
```

**使用说明**：
- 仅用于 SFT 阶段训练
- ⚠️ 许可协议限制：仅限非商业用途
- 建议采样比例：100%（全量使用）

---

## 🔢 中文数学数据集

### 9. Ape210K

**中文小学数学应用题大规模数据集**。

| 属性 | 说明 |
|------|------|
| **数据集名称** | Ape210K: A Large-Scale Chinese Elementary School Math Word Problem Dataset |
| **语言** | 中文 |
| **数据规模** | 20,000 条 |
| **数据类型** | 纯文本应用题 |
| **题目类型** | 应用题（含方程式和答案） |
| **来源平台** | HuggingFace |
| **仓库地址** | [MU-NLPC/Calc-ape210k](https://huggingface.co/datasets/MU-NLPC/Calc-ape210k) |
| **许可协议** | MIT License |
| **论文引用** | [Ape210K: A Large-Scale Chinese Elementary School Math Word Problem Dataset](https://arxiv.org/abs/2012.07201) |
| **发布时间** | 2020年 |

**数据特点**：
- 大规模小学数学应用题
- 包含详细的解题方程式
- 适合训练数学应用能力

**下载方式**：
```bash
python scripts/convert_edu_data.py --dataset ape210k
```

**使用说明**：
- 仅用于 SFT 阶段训练
- 建议采样比例：100%（全量使用）

---

### 10. OpenR1-Math K12

**中文 K12 数学推理链数据集**，包含详细的 CoT（思维链）推理过程。

| 属性 | 说明 |
|------|------|
| **数据集名称** | OpenR1-Math-cn_k12-91k |
| **语言** | 中文 |
| **数据规模** | 20,000 条 |
| **数据类型** | 纯文本数学题 + CoT 推理 |
| **教育阶段** | K12（小学到高中） |
| **来源平台** | HuggingFace |
| **仓库地址** | [Neelectric/OpenR1-Math-cn_k12-91k](https://huggingface.co/datasets/Neelectric/OpenR1-Math-cn_k12-91k) |
| **许可协议** | Apache License 2.0 |
| **发布时间** | 2024年 |

**数据特点**：
- 包含详细的推理步骤（Chain-of-Thought）
- 覆盖小学到高中全部知识点
- 适合训练模型的数学推理能力

**下载方式**：
```bash
python scripts/convert_edu_data.py --dataset openr1_math
```

**使用说明**：
- 仅用于 SFT 阶段训练
- 建议采样比例：100%（全量使用）
- 核心数据集，用于提升中文数学能力

---

### 11. Gaokao MathQA

**高考数学选择题数据集**。

| 属性 | 说明 |
|------|------|
| **数据集名称** | Gaokao Math Multiple Choice Questions |
| **语言** | 中文 |
| **数据规模** | 351 条 |
| **数据类型** | 纯文本选择题 |
| **考试类型** | 高考数学选择题 |
| **来源平台** | HuggingFace |
| **仓库地址** | [hails/agieval-gaokao-mathqa](https://huggingface.co/datasets/hails/agieval-gaokao-mathqa) |
| **许可协议** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **论文引用** | [AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models](https://arxiv.org/abs/2304.06364) |
| **发布时间** | 2023年 |

**数据特点**：
- 来自真实的高考数学试卷
- 涵盖代数、几何、概率等知识点
- 适合评估模型的高中数学能力

**下载方式**：
```bash
python scripts/convert_edu_data.py --dataset gaokao_mathqa
```

**使用说明**：
- 仅用于 SFT 阶段训练
- ⚠️ 数据量较小，建议全量使用
- 可用于评估模型的高考水平数学能力

---

### 12. Gaokao MathCloze

**高考数学填空题数据集**。

| 属性 | 说明 |
|------|------|
| **数据集名称** | Gaokao Math Fill-in-the-Blank Questions |
| **语言** | 中文 |
| **数据规模** | 118 条 |
| **数据类型** | 纯文本填空题 |
| **考试类型** | 高考数学填空题 |
| **来源平台** | HuggingFace |
| **仓库地址** | [hails/agieval-gaokao-mathcloze](https://huggingface.co/datasets/hails/agieval-gaokao-mathcloze) |
| **许可协议** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **论文引用** | [AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models](https://arxiv.org/abs/2304.06364) |
| **发布时间** | 2023年 |

**数据特点**：
- 高考数学填空题，比选择题难度更高
- 需要精确计算和理解
- 数据量较小（118 条）

**下载方式**：
```bash
python scripts/convert_edu_data.py --dataset gaokao_mathcloze
```

**使用说明**：
- 仅用于 SFT 阶段训练
- ⚠️ 数据量非常小，建议全量使用并结合其他数据集
- 可用于训练模型的精确计算能力

---

## 📖 语言理解数据集

### 13. RACE

**阅读理解数据集**，包含中英文阅读理解题目。

| 属性 | 说明 |
|------|------|
| **数据集名称** | RACE: ReAding Comprehension Dataset from Examinations |
| **语言** | 英文（原始）/ 中文（翻译版） |
| **数据规模** | 10,000 条 |
| **数据类型** | 纯文本阅读理解 |
| **题目来源** | 中国初高中英语考试 |
| **来源平台** | HuggingFace |
| **仓库地址** | [ehovy/race](https://huggingface.co/datasets/ehovy/race) |
| **许可协议** | MIT License |
| **论文引用** | [RACE: Reading Comprehension Dataset from Examinations](https://arxiv.org/abs/1704.04683) |
| **发布时间** | 2017年 |

**数据特点**：
- 来自真实的考试题目
- 包含长篇文章和多项选择题
- 需要理解文章内容并推理

**数据集划分**：

| 子集 | 数据量 |
|------|--------|
| train | ~87,000 |
| validation | ~4,900 |
| test | ~4,900 |
| **总计** | ~96,800（采样 10,000） |

**下载方式**：
```bash
python scripts/convert_edu_data.py --dataset race
```

**使用说明**：
- 仅用于 SFT 阶段训练
- ⚠️ 数据量较大，建议采样 10,000-20,000 条
- 用于提升语言理解能力

---

## 🛠️ 数据集下载工具

### 一键下载所有数据集

```bash
# 一键下载全部已有数据（已有则跳过）
python download_data.py

# 或使用 convert_edu_data.py 下载指定数据集
python scripts/convert_edu_data.py --dataset all

# 下载单个数据集
python scripts/convert_edu_data.py --dataset scienceqa
python scripts/convert_edu_data.py --dataset ceval
```

### 数据集转换工具

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `convert_edu_data.py` | 通用数据集转换 | HuggingFace datasets | Parquet |
| `convert_windata.py` | windata 转换 | ModelScope dataset | Parquet |
| `resample_data.py` | 数据重采样 | Parquet | Parquet |

### 数据集质量检查

```bash
# 检查数据集完整性
python -c "
import pandas as pd
datasets = [
    'dataset/edu_science.parquet',
    'dataset/edu_windata_math.parquet',
    # ...
]
for ds in datasets:
    df = pd.read_parquet(ds)
    print(f'{ds}: {len(df)} 条, {df.columns.tolist()}')
"
```

---

## ⚖️ 许可协议说明

### 开源许可协议

| 协议 | 数据集 | 商业用途 | 修改要求 | 署名要求 |
|------|--------|:--------:|:--------:|:--------:|
| **CC BY 4.0** | ScienceQA, MathVerse, Gaokao MathQA/Cloze | ✅ 可用 | ✅ 可修改 | ✅ 需要 |
| **CC BY-NC 4.0** | CMMLU | ❌ 禁止 | ✅ 可修改 | ✅ 需要 |
| **CC-BY-4.0** | windata-math | ✅ 可用 | ✅ 可修改 | ✅ 需要 |
| **MIT** | MathVista, ChartQA, Ape210K, RACE | ✅ 可用 | ✅ 可修改 | ❌ 不需要 |
| **Apache 2.0** | OCR-VQA, C-Eval, OpenR1-Math | ✅ 可用 | ✅ 可修改 | ❌ 不需要 |

### 使用建议

1. **商业用途**：优先使用 CC BY 4.0、MIT、Apache 2.0 协议的数据集
2. **学术研究**：可以使用所有数据集
3. **发布模型**：注意遵守各个数据集的许可协议
4. **数据混合**：确保最终产品符合所有数据集的许可要求

---

## 📊 数据集统计信息

### 数据分布统计

```
总数据量: 116,198 条

按语言分布:
  英文:  41,158 条 (35.4%)
  中文:  65,040 条 (56.0%)
  中英:  10,000 条 (8.6%)

按类型分布:
  图文多模态: 51,158 条 (44.0%) ✅ 含图片
  纯文本:     65,040 条 (56.0%) ❌ 无图片

按用途分布:
  SFT 全阶段:  6,218 条 (5.4%) - ScienceQA
  SFT 训练:  109,980 条 (94.6%) - 其他数据集

按学科分布:
  数学:        56,409 条 (48.6%)
  科学:         6,218 条 (5.4%)
  图表/OCR:    30,000 条 (25.8%)
  理科综合:    14,571 条 (12.5%)
  语言理解:    10,000 条 (8.6%)
```

### 数据集大小估算

| 数据集 | Parquet 大小 | 原始图像大小 | 总计 |
|--------|-------------|-------------|------|
| edu_science.parquet | ~100MB | 0 (已内联) | ~100MB |
| edu_windata_math.parquet | ~50MB | 0 (已内联) | ~50MB |
| edu_ocr.parquet | ~200MB | 0 (已内联) | ~200MB |
| edu_*.parquet (其他) | ~1.2GB | 0 | ~1.2GB |
| **总计** | **~1.7GB** | 0 | **~1.7GB** |

> 注：所有图像已转换为 bytes 并内联到 Parquet 文件中，无需额外存储图像文件。

---

## 🔧 常见问题

### Q1: 如何获取新的数据集？

```bash
# 1. 在 HuggingFace 上搜索相关数据集
# 2. 使用 scripts/convert_edu_data.py 转换格式
python scripts/convert_edu_data.py --dataset new_dataset

# 3. 添加到训练配置
# 编辑 train_sft.py 的 --data_paths 参数
```

### Q2: 数据集下载失败怎么办？

```bash
# 方法 1: 检查网络连接
ping huggingface.co

# 方法 2: 使用镜像站点
export HF_ENDPOINT=https://hf-mirror.com

# 方法 3: 手动下载后放置到正确位置
# 下载 .parquet 文件后放置到 dataset/ 目录
```

### Q3: 如何清理不再需要的数据集？

```bash
# 删除不需要的 Parquet 文件
rm dataset/edu_unused.parquet

# 或删除原始数据（如果有）
rm -rf dataset/unused-raw-data/
```

### Q4: 如何验证数据集质量？

```bash
# 使用 analyze_datasets.py（如果存在）
python scripts/analyze_datasets.py

# 或手动检查
python -c "
import pandas as pd
df = pd.read_parquet('dataset/edu_science.parquet')
print(f'总条数: {len(df)}')
print(f'列: {df.columns.tolist()}')
print(f'缺失值:\n{df.isnull().sum()}')
"
```

---

## 📚 参考资源

### 核心论文

1. **ScienceQA**: [ScienceQA: Multimodal Science Question Answering](https://arxiv.org/abs/2210.11431)
2. **MathVista**: [MathVista: Mathematical Visual Reasoning Benchmark](https://arxiv.org/abs/2308.06358)
3. **C-Eval**: [C-Eval: A Multi-level Multi-discipline Chinese Evaluation Benchmark](https://arxiv.org/abs/2305.08322)
4. **CMMLU**: [CMMLU: A Massive Chinese Multi-modal Language Understanding Evaluation Benchmark](https://arxiv.org/abs/2304.12547)
5. **RACE**: [RACE: Reading Comprehension Dataset from Examinations](https://arxiv.org/abs/1704.04683)

### 相关工具

- **HuggingFace Datasets**: https://huggingface.co/docs/datasets
- **ModelScope SDK**: https://modelscope.cn/docs
- **Parquet Format**: https://parquet.apache.org/

---

## 📝 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2026-06-05 | v1.0 | 初始版本，包含 13 个数据集 |
| 2026-06-05 | v1.1 | 添加 windata-math 数据集来源 |
| 2026-06-05 | v1.2 | 完善许可协议说明和使用指南 |

---

> 📅 最后更新: 2026-06-05  
> 🤖 维护者: QwenSearch Team  
> 📧 如有问题，请提交 Issue