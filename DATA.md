# QwenSearch 数据集全览

## 数据集清单（6 个已下载，共 45,729 条）

| # | 数据集 | 文件 | 条数 | 语言 | 类型 | 学科 | 学段 |
|---|--------|------|------|------|------|------|------|
| 1 | **ScienceQA** | `edu_science.parquet` | 6,218 | EN | 含图 | 全理科 | 小学-高中 |
| 2 | **MathVerse** | `edu_math_verse.parquet` | 3,940 | EN | 含图 | 数学 | 初中-大学 |
| 3 | **MathVista** | `edu_math_vista.parquet` | 1,000 | EN | 含图 | 数学 | 小学-大学 |
| 4 | **OCR-VQA** | `edu_ocr.parquet` | 20,000 | EN | 含图 | 文字识别 | 通用 |
| 5 | **C-Eval** | `edu_ceval.parquet` | 2,654 | CN | 纯文本 | 14个理科 | 初中-大学 |
| 6 | **CMMLU** | `edu_cmmlu.parquet` | 11,917 | CN | 纯文本 | 67个学科 | 中学-大学 |
| | **合计** | | **45,729** | | | | |

## 各训练阶段使用哪些数据

```
┌──────────────────────────────────────────────────────────────────┐
│                      SFT 阶段 (全量 45,729 条)                     │
│                                                                  │
│  ┌──────────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐           │
│  │ScienceQA │ │MathVerse  │ │MathVista  │ │ OCR-VQA  │           │
│  │  6,218   │ │  3,940    │ │  1,000    │ │ 20,000   │           │
│  │ 全理科图 │ │ 数学图    │ │ 数学图    │ │ OCR图    │           │
│  └──────────┘ └───────────┘ └───────────┘ └──────────┘           │
│                                                         ⚠️降权  │
│  ┌──────────┐ ┌───────────┐                                     │
│  │ C-Eval   │ │  CMMLU    │   OCR-VQA 占比 20-30% batch         │
│  │  2,654   │ │  11,917   │                                     │
│  │ 中文理科 │ │ 中文综合  │                                     │
│  └──────────┘ └───────────┘                                     │
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

## 评估用哪些数据（与训练数据隔离）

```
┌───────────────────────────────────────────────────────────────────┐
│                          评估测试集                                │
│                                                                   │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐    │
│  │  ScienceQA test     │  │  C-Eval holdout (5个保留学科)     │    │
│  │  test split 4,241   │  │  high_school_math/physics         │    │
│  │  ⚠️ 仅 --stage full  │  │  college_physics/programming      │    │
│  │  使用（最终holdout） │  │  discrete_math                   │    │
│  └─────────────────────┘  └──────────────────────────────────┘    │
│                                                                   │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐    │
│  │  ScienceQA val      │  │  自定义 Parquet 数据              │    │
│  │  validation 4,241   │  │  本地 edu_science.parquet         │    │
│  │  日常迭代使用 500    │  │  日常迭代使用 200                 │    │
│  └─────────────────────┘  └──────────────────────────────────┘    │
│                                                                   │
│  ⚠️ C-Eval 训练集 ≠ 评估集：训练用了 9 个理科子集，                  │
│  评估用另外 5 个保留学科，确保评估公正                                │
└───────────────────────────────────────────────────────────────────┘
```

## 下载命令

```bash
# 一键下载全部 6 个数据集
python download_data.py

# 单个下载示例
python scripts/convert_edu_data.py --dataset scienceqa --output dataset/edu_science.parquet
python scripts/convert_edu_data.py --dataset ceval --output dataset/edu_ceval.parquet

# 验证数据集
python -c "
import pyarrow.parquet as pq, os
for f in ['dataset/edu_science.parquet','dataset/edu_math_verse.parquet','dataset/edu_math_vista.parquet','dataset/edu_ocr.parquet','dataset/edu_ceval.parquet','dataset/edu_cmmlu.parquet']:
    print(f'{f}: {pq.ParquetFile(f).metadata.num_rows} 条') if os.path.exists(f) else print(f'{f}: 不存在')
"
```

## 待补充的数据方向

| 缺口 | 优先级 | 补充方案 |
|------|--------|----------|
| 中文图文几何题 | 🔴 高 | GeoQA+ GitHub 手动下载 → 本地转换 |
| 中文教材截图 | 🔴 高 | 爬取初高中教材PDF → 截图 → 标注 |
| 代长解析数学题 | 🟡 中 | GPT-4o 为 C-Eval 补充详细解析 |
| 函数图/实验图 | 🟢 低 | MathVista 翻译增强 |