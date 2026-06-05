# QwenSearch 数据集全览

## 数据集清单（10 个已下载，85,729 条）

### 已下载生效（8 个，73,812 条）

| # | 数据集 | 文件 | 条数 | 语言 | 类型 | 学科 | HF 来源 |
|---|--------|------|------|------|------|------|----------|
| 1 | **OCR-VQA** | `edu_ocr.parquet` | 20,000 | EN | 含图 | OCR 文字识别 | `MMInstruction/OCR-VQA` |
| 2 | **Ape210K** | `edu_ape210k.parquet` | 20,000 | CN | 纯文本 | 小学数学 | `MU-NLPC/Calc-ape210k` |
| 3 | **CMMLU** | `edu_cmmlu.parquet` | 11,917 | CN | 纯文本 | 67 学科综合 | `haonan-li/CMMLU` |
| 4 | **ChartQA** | `edu_chartqa.parquet` | 10,000 | EN | 含图 | 图表理解 | `HuggingFaceM4/ChartQA` |
| 5 | **RACE** | `edu_race.parquet` | 10,000 | CN/EN | 纯文本 | 阅读理解 | `ehovy/race` |
| 6 | **ScienceQA** | `edu_science.parquet` | 6,218 | EN | 含图 | 全理科 | `derek-thomas/ScienceQA` |
| 7 | **MathVerse** | `edu_math_verse.parquet` | 3,940 | EN | 含图 | 数学 | `AI4Math/MathVerse` |
| 8 | **C-Eval** | `edu_ceval.parquet` | 2,654 | CN | 纯文本 | 14 理科 | `ceval/ceval-exam` |
| | **小计** | | **73,812** | | | | |

### 已下载但需修复（1 个，11,917 条）

| # | 数据集 | 文件 | 条数 | 问题 | 解决方案 |
|---|--------|------|------|------|----------|
| 9 | **BioVQA** | `edu_biology.parquet` | 0 | 仅表头无数据 | 代码待修复（HF 镜像 `MMInstruction/BioVQA` 已下架）|

### 已定位但下载受阻（4 个，~21K 条，因网络问题）

| # | 数据集 | 文件 | 预期条数 | HF 修复路径 | 状态 |
|---|--------|------|:--------:|------|:----:|
| 10 | **MMMU** | `edu_mmmu.parquet` | ~5K | `MMMU/MMMU`（30 学科 config） | 代码已修复，SSL 中断 |
| 11 | **GeoQA+** | `edu_geoqa.parquet` | ~8K | `leonardPKU/GEOQA_R1V_Train_8K` | 代码已修复，SSL 中断 |
| 12 | **Geometry3K** | `edu_geometry3k.parquet` | ~3K | `hiyouga/geometry3k` | 代码已修复，SSL 中断 |
| 13 | **DVQA** | `edu_dvqa.parquet` | ~5K | `DavidNguyen/DVQA` | 代码已修复，SSL 中断 |

### HF 已下架无法恢复（7 个）

| # | 数据集 | 原因 |
|---|--------|------|
| 14 | **TabMWP** | HF 404 下架 |
| 15 | **GAOKAO-Bench** | HF 404 下架 |
| 16 | **Math23K** | HF 404 下架 |
| 17 | **AI2D** | 需 gated access |
| 18 | **TQA** | 需 gated access |
| 19 | **VizWiz** | 需 gated access |
| 20 | **CLEVR-Math** | datasets 库不兼容旧格式 |

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
# 设置 HuggingFace Token（如有）
set HF_TOKEN=hf_xxx

# 一键下载全部已有数据（已有则跳过）
python download_data.py

# ⚠️ 如遇 SSL 错误，可尝试：
# 方法1：设置 HF 代理
set HF_ENDPOINT=https://hf-mirror.com
# 方法2：逐个下载避免超时
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
| 中文图文几何题 | 🔴 高 | GeoQA+ 已定位 `leonardPKU/GEOQA_R1V_Train_8K`，需稳定网络下载 |
| 大学级多模态题 | 🔴 高 | MMMU 已修复 30 学科 config 遍历，需稳定网络下载 |
| 几何图文解析 | 🟡 中 | Geometry3K 已定位 `hiyouga/geometry3k`，需稳定网络下载 |
| 信息图问答 | 🟡 中 | DVQA 已定位 `DavidNguyen/DVQA`，需稳定网络下载 |
| 生物图文题 | 🟡 中 | BioVQA 需寻找新的 HF 镜像 |
| 表格数学应用题 | 🟡 中 | TabMWP 可从原始 GitHub 仓库手动下载 |
| 中文教材截图 | 🟡 中 | 爬取初高中教材PDF → 截图 → 标注 |
| 代长解析数学题 | 🟡 中 | GPT-4o 为 C-Eval 补充详细解析 |