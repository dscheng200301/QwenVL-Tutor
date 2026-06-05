# QwenSearch 数据集全览

## 数据集清单（12 个已下载，106,198 条）

| # | 数据集 | 文件 | 条数 | 语言 | 类型 | 学科 | HF 来源 |
|---|--------|------|------|------|------|------|----------|
| 1 | **OCR-VQA** | `edu_ocr.parquet` | 20,000 | EN | 含图 | OCR 文字识别 | `MMInstruction/OCR-VQA` |
| 2 | **Ape210K** | `edu_ape210k.parquet` | 20,000 | CN | 纯文本 | 小学数学 | `MU-NLPC/Calc-ape210k` |
| 3 | **OpenR1-Math CN K12** | `edu_openr1_math.parquet` | 20,000 | CN | 纯文本 | K12 数学 | `Neelectric/OpenR1-Math-cn_k12-91k` |
| 4 | **CMMLU** | `edu_cmmlu.parquet` | 11,917 | CN | 纯文本 | 67 学科综合 | `haonan-li/CMMLU` |
| 5 | **ChartQA** | `edu_chartqa.parquet` | 10,000 | EN | 含图 | 图表理解 | `HuggingFaceM4/ChartQA` |
| 6 | **RACE** | `edu_race.parquet` | 10,000 | CN/EN | 纯文本 | 阅读理解 | `ehovy/race` |
| 7 | **ScienceQA** | `edu_science.parquet` | 6,218 | EN | 含图 | 全理科 | `derek-thomas/ScienceQA` |
| 8 | **MathVerse** | `edu_math_verse.parquet` | 3,940 | EN | 含图 | 数学 | `AI4Math/MathVerse` |
| 9 | **C-Eval** | `edu_ceval.parquet` | 2,654 | CN | 纯文本 | 14 理科 | `ceval/ceval-exam` |
| 10 | **MathVista** | `edu_math_vista.parquet` | 1,000 | EN | 含图 | 数学 | `AI4Math/MathVista` |
| 11 | **Gaokao MathQA** | `edu_gaokao_mathqa.parquet` | 351 | CN | 纯文本 | 高考数学 | `hails/agieval-gaokao-mathqa` |
| 12 | **Gaokao MathCloze** | `edu_gaokao_mathcloze.parquet` | 118 | CN | 纯文本 | 高考数学 | `hails/agieval-gaokao-mathcloze` |
| | **合计** | | **106,198** | | | | |

## 磁盘占用

| 文件 | 大小 | 条数 |
|------|------|------|
| `edu_ocr.parquet` | 957.7 MB | 20,000 |
| `edu_chartqa.parquet` | 375.2 MB | 10,000 |
| `edu_science.parquet` | 160.2 MB | 6,218 |
| `edu_math_verse.parquet` | 108.8 MB | 3,940 |
| `edu_math_vista.parquet` | 58.0 MB | 1,000 |
| `edu_openr1_math.parquet` | 10.7 MB | 20,000 |
| `edu_ape210k.parquet` | 3.3 MB | 20,000 |
| `edu_race.parquet` | 2.1 MB | 10,000 |
| `edu_ceval.parquet` | 0.5 MB | 2,654 |
| `edu_gaokao_mathqa.parquet` | 0.1 MB | 351 |
| `edu_cmmlu.parquet` | ~0 MB | 11,917 |
| `edu_gaokao_mathcloze.parquet` | ~0 MB | 118 |
| **合计** | **~1,677 MB** | **106,198** |

## 下载命令

```bash
# 一键下载全部已有数据（已有则跳过）
python download_data.py

# 验证数据集
python -c "
import pyarrow.parquet as pq, os, glob
for f in sorted(glob.glob('dataset/edu_*.parquet')):
    size = os.path.getsize(f) / 1024 / 1024
    rows = pq.ParquetFile(f).metadata.num_rows
    print(f'{f}: {rows:>6} 条 ({size:.1f} MB)')
"
```

## 各训练阶段使用哪些数据

```
┌──────────────────────────────────────────────────────────────────┐
│                      SFT 阶段 (全量 106,198 条)                    │
│                                                                  │
│  ┌──────────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐           │
│  │ScienceQA │ │MathVerse  │ │MathVista  │ │ OCR-VQA  │           │
│  │  6,218   │ │  3,940    │ │  1,000    │ │ 20,000   │           │
│  │ 全理科图 │ │ 数学图    │ │ 数学图    │ │ OCR图    │           │
│  └──────────┘ └───────────┘ └───────────┘ └──────────┘           │
│                                                         ⚠️降权  │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐           │
│  │ C-Eval   │ │  CMMLU    │ │ ChartQA  │ │ Ape210K  │           │
│  │  2,654   │ │  11,917   │ │ 10,000   │ │ 20,000   │           │
│  │ 中文理科 │ │ 中文综合  │ │ 图表理解 │ │ 小学数学 │           │
│  └──────────┘ └───────────┘ └──────────┘ └──────────┘           │
│  ┌──────────┐ ┌───────────┐ ┌───────────┐                       │
│  │  RACE    │ │OpenR1-Math│ │GaokaoMath │  🆕 新增中文数学      │
│  │ 10,000   │ │  20,000   │ │   469     │  OpenR1 含完整 CoT    │
│  │ 阅读理解 │ │ 含 CoT    │ │ 高考真题  │                       │
│  └──────────┘ └───────────┘ └───────────┘                       │
│                                                                  │
│  EduDataset → question + answer pairs → cross-entropy loss       │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      DPO 阶段 (精选 ScienceQA)                    │
│                                                                  │
│  ┌──────────┐                                                    │
│  │ScienceQA │  完整解析答案 → chosen (完整回答)                    │
│  │  6,218   │  截断到 1/3 → rejected (简短回答)                   │
│  └──────────┘                                                    │
│                                                                  │
│  EduDPODataset → 偏好对 → DPO loss (log-sigmoid)                 │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                     GRPO 阶段 (精选 ScienceQA)                    │
│                                                                  │
│  ┌──────────┐                                                    │
│  │ScienceQA │  prompt 部分 → 生成 K=4 候选                         │
│  │  6,218   │  GT答案 → EduRewardModel 五维度评分                  │
│  └──────────┘                                                    │
│                                                                  │
│  EduGRPODataset → prompt + GT → GRPO PPO-clip loss               │
└──────────────────────────────────────────────────────────────────┘
```

## 评估数据集（12 个 holdout，与训练数据隔离）

| # | 评估数据集 | 文件 | 条数 | 来源 | 评估指标 |
|---|-----------|------|------|------|----------|
| 1 | **ScienceQA** | `eval/eval_science.parquet` | 932 | 训练集 15% holdout | 答案准确率 + 步骤完整率 |
| 2 | **C-Eval** | HF 动态加载 | 500 | 5 个保留学科 | 选项匹配率 |
| 3 | **OCR-VQA** | `eval/eval_ocr.parquet` | 1,000 | 训练集 15% holdout | 关键词匹配率 |
| 4 | **Ape210K** | `eval/eval_ape210k.parquet` | 1,000 | 训练集 15% holdout | 关键词匹配率 |
| 5 | **OpenR1-Math** | `eval/eval_openr1_math.parquet` | 1,000 | 训练集 15% holdout | 关键词匹配率 + 步骤完整率 |
| 6 | **ChartQA** | `eval/eval_chartqa.parquet` | 1,000 | 训练集 15% holdout | 答案匹配率 |
| 7 | **CMMLU** | `eval/eval_cmmlu.parquet` | 1,000 | 训练集 15% holdout | 选项匹配率 |
| 8 | **MathVerse** | `eval/eval_math_verse.parquet` | 591 | 训练集 15% holdout | 关键词匹配率 |
| 9 | **MathVista** | `eval/eval_math_vista.parquet` | 150 | 训练集 15% holdout | 答案匹配率 |
| 10 | **RACE** | `eval/eval_race.parquet` | 1,000 | 训练集 15% holdout | 选项匹配率 |
| 11 | **Gaokao MathQA** | `eval/eval_gaokao_mathqa.parquet` | 351 | 全量 holdout | 选项匹配率 |
| 12 | **Gaokao MathCloze** | `eval/eval_gaokao_mathcloze.parquet` | 118 | 全量 holdout | 数值匹配率 |

### 评估命令

```bash
# 评估单个数据集
python eval_edu.py --model_path out/edu_sft --eval_dataset openr1_math --max_samples 500
python eval_edu.py --model_path out/edu_sft --eval_dataset chartqa --max_samples 200

# 一键评估所有数据集
python eval_edu.py --model_path out/edu_sft --eval_all --max_samples 200

# 传统分阶段评估（兼容原有流程）
python eval_edu.py --stage sft   # ScienceQA + C-Eval + 自定义
python eval_edu.py --stage full  # 全量最终评估
```

### 评估指标说明

| 指标 | 计算方法 | 适用数据集 |
|------|----------|-----------|
| **答案准确率** | 检查模型回复中是否包含正确答案 | ScienceQA, MathVista, Gaokao |
| **选项匹配率** | 检查答案选项字母是否出现 | C-Eval, CMMLU, RACE |
| **关键词匹配率** | GT 答案中关键词与回复的交集比率 | OCR-VQA, Ape210K, MathVerse, OpenR1-Math |
| **步骤完整率** | 回复中是否包含分步推理关键词 | 所有图文数学数据集 |
| **启发式引导率** | 回复中是否包含引导性语言 | 亲子教育场景评估 |

## 待补充的数据方向

| 缺口 | 优先级 | 补充方案 |
|------|--------|----------|
| 中文图文几何题 | 🔴 高 | GeoQA+ 已定位 `leonardPKU/GEOQA_R1V_Train_8K` |
| 大学级多模态题 | 🔴 高 | MMMU 已修复 30 学科 config 遍历 |
| 几何图文解析 | 🟡 中 | Geometry3K 已定位 `hiyouga/geometry3k` |
| 信息图问答 | 🟡 中 | DVQA 已定位 `DavidNguyen/DVQA` |
| 中文教材截图 | 🟡 中 | 爬取初高中教材PDF → 截图 → 标注 |