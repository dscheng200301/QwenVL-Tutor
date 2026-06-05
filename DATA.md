# QwenSearch 数据集全览

## 数据集清单（共 10 个已下载，~86K 条）

### 已下载（10 个，85,729 条）

| # | 数据集 | 文件 | 条数 | 语言 | 类型 | 学科 | 来源 |
|---|--------|------|------|------|------|------|------|
| 1 | **ScienceQA** | `edu_science.parquet` | 6,218 | EN | 含图 | 全理科 | `derek-thomas/ScienceQA` |
| 2 | **MathVerse** | `edu_math_verse.parquet` | 3,940 | EN | 含图 | 数学 | `AI4Math/MathVerse` |
| 3 | **MathVista** | `edu_math_vista.parquet` | 1,000 | EN | 含图 | 数学 | `AI4Math/MathVista` |
| 4 | **OCR-VQA** | `edu_ocr.parquet` | 20,000 | EN | 含图 | 文字识别 | `MMInstruction/OCR-VQA` |
| 5 | **C-Eval** | `edu_ceval.parquet` | 2,654 | CN | 纯文本 | 14理科 | `ceval/ceval-exam` |
| 6 | **CMMLU** | `edu_cmmlu.parquet` | 11,917 | CN | 纯文本 | 67学科 | `haonan-li/CMMLU` |
| 7 | **ChartQA** | `edu_chartqa.parquet` | 10,000 | EN | 含图 | 图表理解 | `HuggingFaceM4/ChartQA` |
| 8 | **RACE** | `edu_race.parquet` | 10,000 | CN/EN | 纯文本 | 阅读理解 | `ehovy/race` |
| 9 | **Ape210K** | `edu_ape210k.parquet` | 20,000 | CN | 纯文本 | 小学数学 | `MU-NLPC/Calc-ape210k` |
| 10 | **BioVQA** | `edu_biology.parquet` | 0 | EN | 含图 | 生物 | 待修复 |
| | **合计** | | **85,729** | | | | |

### 待修复可下载（4 个，~20K 条等待下载）

| # | 数据集 | 文件 | 预期条数 | 语言 | 含图 | 问题 | 修复方案 |
|---|--------|------|:--------:|:----:|:----:|------|----------|
| 11 | **MMMU** | `edu_mmmu.parquet` | ~5K | EN | ✅ | Config 缺失 | 遍历 30 学科 config（已修复代码） |
| 12 | **GeoQA+** | `edu_geoqa.parquet` | ~8K | CN | ✅ | 旧镜像失效 | `leonardPKU/GEOQA_R1V_Train_8K`（已修复代码） |
| 13 | **Geometry3K** | `edu_geometry3k.parquet` | ~3K | EN | ✅ | 旧镜像失效 | `hiyouga/geometry3k`（已修复代码） |
| 14 | **DVQA** | `edu_dvqa.parquet` | ~5K | EN | ✅ | 旧镜像失效 | `DavidNguyen/DVQA`（已修复代码） |

### 短期待修复（2 个）

| # | 数据集 | 原因 | 可能解决方案 |
|---|--------|------|-------------|
| 15 | **CLEVR-Math** | `datasets` 库不支持旧 `.py` 格式 | 降级 `datasets` 到旧版本 |
| 16 | **GAOKAO-Bench** | HF 已下架（404） | `RUCAIBox/gaokao-bench` 需申请 gated access |

### 长期待补充（5 个，HF 已下架）

| # | 数据集 | 原因 |
|---|--------|------|
| 17 | **TabMWP** | HF 已下架 |
| 18 | **Math23K** | HF 已下架 |
| 19 | **AI2D** | 需 gated access |
| 20 | **TQA** | 需 gated access |
| 21 | **VizWiz** | 需 gated access |

## 磁盘占用

| 文件 | 大小 | 条数 |
|------|------|------|
| `edu_ocr.parquet` | 957.7 MB | 20,000 |
| `edu_chartqa.parquet` | 375.2 MB | 10,000 |
| `edu_science.parquet` | 160.2 MB | 6,218 |
| `edu_math_verse.parquet` | 108.8 MB | 3,940 |
| `edu_math_vista.parquet` | 58.0 MB | 1,000 |
| `edu_ape210k.parquet` | 3.3 MB | 20,000 |
| `edu_race.parquet` | 2.1 MB | 10,000 |
| `edu_ceval.parquet` | 0.5 MB | 2,654 |
| `edu_cmmlu.parquet` | ~0 MB | 11,917 |
| `edu_biology.parquet` | ~0 MB | 0 |
| **合计** | **~1,666 MB** | **85,729** |

## 下载命令

```bash
# 设置 HuggingFace Token（必须）
set HF_TOKEN=hf_xxx

# 一键下载全部 21 个数据集（已有则跳过）
python download_data.py

# 单个下载示例
python scripts/convert_edu_data.py --dataset ape210k --output dataset/edu_ape210k.parquet
python scripts/convert_edu_data.py --dataset mmmu --output dataset/edu_mmmu.parquet --max_samples 5000
python scripts/convert_edu_data.py --dataset geoqa --output dataset/edu_geoqa.parquet --max_samples 10000
python scripts/convert_edu_data.py --dataset geometry3k --output dataset/edu_geometry3k.parquet --max_samples 3000
python scripts/convert_edu_data.py --dataset dvqa --output dataset/edu_dvqa.parquet --max_samples 5000

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
│                      SFT 阶段 (全量 85,729 条)                     │
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
│  ┌──────────┐                                                    │
│  │  RACE    │   OCR-VQA 占比 20-30% batch                       │
│  │ 10,000   │                                                    │
│  │ 阅读理解 │                                                    │
│  └──────────┘                                                    │
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

## 待补充的数据方向

| 缺口 | 优先级 | 补充方案 |
|------|--------|----------|
| 中文图文几何题 | 🔴 高 | GeoQA+ 已定位新路径，待下载 |
| 大学级多模态题 | 🔴 高 | MMMU 已修复 config 遍历，待下载 |
| 几何图文解析 | 🟡 中 | Geometry3K 已定位新路径，待下载 |
| 信息图问答 | 🟡 中 | DVQA 已定位新路径，待下载 |
| 中文教材截图 | 🟡 中 | 爬取初高中教材PDF → 截图 → 标注 |
| 代长解析数学题 | 🟡 中 | GPT-4o 为 C-Eval 补充详细解析 |
| 函数图/实验图 | 🟢 低 | MathVista 翻译增强 |