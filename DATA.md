# QwenSearch 数据集全览

## 数据集清单（23 个支持，~200K+ 条）

### 已下载（6 个，45,729 条）

| # | 数据集 | 文件 | 条数 | 语言 | 类型 | 学科 |
|---|--------|------|------|------|------|------|
| 1 | **ScienceQA** | `edu_science.parquet` | 6,218 | EN | 含图 | 全理科 |
| 2 | **MathVerse** | `edu_math_verse.parquet` | 3,940 | EN | 含图 | 数学 |
| 3 | **MathVista** | `edu_math_vista.parquet` | 1,000 | EN | 含图 | 数学 |
| 4 | **OCR-VQA** | `edu_ocr.parquet` | 20,000 | EN | 含图 | 文字识别 |
| 5 | **C-Eval** | `edu_ceval.parquet` | 2,654 | CN | 纯文本 | 14理科 |
| 6 | **CMMLU** | `edu_cmmlu.parquet` | 11,917 | CN | 纯文本 | 67学科 |
| | **合计** | | **45,729** | | | |

### 新增可下载（17 个，~150K+ 条限样）

| # | 数据集 | 文件 | 限样 | 语言 | 含图 | 说明 |
|---|--------|------|:----:|:----:|:----:|------|
| 7 | **TabMWP** | `edu_tabmwp.parquet` | 全量38K | EN | ✅ | 表格数学应用题 |
| 8 | **GAOKAO-Bench** | `edu_gaokao.parquet` | 全量6K | CN | ❌ | 高考真题 |
| 9 | **Math23K** | `edu_math23k.parquet` | 20K | CN | ❌ | 中文小学数学 |
| 10 | **Ape210K** | `edu_ape210k.parquet` | 20K | CN | ❌ | 中文小学数学大规模 |
| 11 | **MMMU** | `edu_mmmu.parquet` | 5K | EN | ✅ | 大学级多模态 |
| 12 | **GeoQA+** | `edu_geoqa.parquet` | 10K | CN | ✅ | 中文几何 |
| 13 | **BioVQA** | `edu_biology.parquet` | 5K | EN | ✅ | 生物图文 |
| 14 | **Geometry3K** | `edu_geometry3k.parquet` | 3K | EN | ✅ | 几何图文解析 |
| 15 | **ChartQA** | `edu_chartqa.parquet` | 10K | EN | ✅ | 柱状折线饼图 |
| 16 | **DVQA** | `edu_dvqa.parquet` | 5K | EN | ✅ | 信息图问答 |
| 17 | **AI2D** | `edu_ai2d.parquet` | 4K | EN | ✅ | 科学示意图 |
| 18 | **VizWiz** | `edu_vizwiz.parquet` | 5K | EN | ✅ | 真实模糊场景 |
| 19 | **TQA** | `edu_tqa.parquet` | 5K | EN | ✅ | 教科书图文 |
| 20 | **CLEVR-Math** | `edu_clevr_math.parquet` | 5K | EN | ✅ | 合成图推理 |
| 21 | **RACE** | `edu_race.parquet` | 10K | CN/EN | ❌ | 中英文阅读 |
| 22 | **Ape210K(extra)** | `edu_ape210k_extra.parquet` | — | CN | ❌ | 剩余190K |
| 23 | **OCR-VQA(extra)** | `edu_ocr_extra.parquet` | — | EN | ✅ | 剩余180K |

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
# 一键下载全部 23 个数据集（已有则跳过）
python download_data.py

# 单个下载示例
python scripts/convert_edu_data.py --dataset geometry3k --output dataset/edu_geometry3k.parquet
python scripts/convert_edu_data.py --dataset chartqa --output dataset/edu_chartqa.parquet
python scripts/convert_edu_data.py --dataset ai2d --output dataset/edu_ai2d.parquet

# 验证数据集
python -c "
import pyarrow.parquet as pq, os, glob
for f in sorted(glob.glob('dataset/edu_*.parquet')):
    size = os.path.getsize(f) / 1024 / 1024
    print(f'{f}: {pq.ParquetFile(f).metadata.num_rows:>6} 条 ({size:.1f} MB)')
"
```

## 待补充的数据方向

| 缺口 | 优先级 | 补充方案 |
|------|--------|----------|
| 中文图文几何题 | 🔴 高 | GeoQA+ GitHub 手动下载 → 本地转换 |
| 中文教材截图 | 🔴 高 | 爬取初高中教材PDF → 截图 → 标注 |
| 代长解析数学题 | 🟡 中 | GPT-4o 为 C-Eval 补充详细解析 |
| 函数图/实验图 | 🟢 低 | MathVista 翻译增强 |