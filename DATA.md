# 📚 QwenVL-Tutor 数据集说明

> 22 个训练数据集 + 19 个评估数据集 | 评估集与训练集严格分离

---

## 🚀 快速开始

```bash
# 一键下载所有数据集（自动分离 eval / train）
python download_all_data.py

# 验证：检查训练集和评估集是否重叠（应为 0）
python -c "
import pyarrow.parquet as pq
t = pq.read_table('dataset/edu_science.parquet').to_pylist()
e = pq.read_table('dataset/eval/eval_science.parquet').to_pylist()
print('重叠:', len(set(r.get('question','')[:100] for r in t) & set(r.get('question','')[:100] for r in e)))"
```

**目录结构**：

```
dataset/
├── edu_*.parquet              # 训练集（不含评估样本）
├── eval/
│   └── eval_*.parquet         # 评估集（与训练集严格分离）
└── .gitkeep
```

---

## 📊 数据集清单（22 训练 / 19 评估）

> **图标说明**：🔵 训练 | 🟢 评估 | 📦 规模 | 🔗 HuggingFace 仓库 | ⚖️ 许可证

### 🇨🇳 中文核心图文数学（3 个）

| 数据集 | 来源 | 训练 | 评估 | 许可 |
|--------|------|:----:|:----:|------|
| **We-Math 2.0** | [FLAGOPEN](https://github.com/FLAGOPEN/We-Math) | 10K | 500 | Apache 2.0 |
| **Geo170K** | [Luckyjhg/Geo170K](https://huggingface.co/datasets/Luckyjhg/Geo170K) | 50K | 500 | CC BY-NC 4.0 |
| **windata-math** | [Lin-Chen/MMStar](https://huggingface.co/datasets/PKU-Alignment/windata) | 10K | 500 | Apache 2.0 |

### 🇨🇳 中文多学科图文做题（4 个，🆕）

| 数据集 | 来源 | 训练 | 评估 | 许可 |
|--------|------|:----:|:----:|------|
| **CMMU** | [FlagOpen/CMMU](https://github.com/FlagOpen/CMMU) | 1.8K | 200 | Apache 2.0 |
| **CMMMU** | [m-a-p/CMMMU](https://huggingface.co/datasets/m-a-p/CMMMU) | 3K | 200 | CC BY-NC-SA 4.0 |
| **M3Exam** | [DAMO-NLP-SG/M3Exam](https://github.com/DAMO-NLP-SG/M3Exam) | 3K | 200 | CC BY-SA 4.0 |
| **MMSciBench** | [XinwuYe/MMSciBench](https://huggingface.co/datasets/XinwuYe/MMSciBench) | 1K | 100 | MIT |

### 🔬 核心图文数学（3 个）

| 数据集 | 来源 | 训练 | 评估 | 许可 |
|--------|------|:----:|:----:|------|
| **ScienceQA** | [derek-thomas/ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) | 6.2K | 932 | CC BY-NC 4.0 |
| **MathVerse** | [AI4Math/MathVerse](https://huggingface.co/datasets/AI4Math/MathVerse) | 3.9K | 591 | CC BY-NC 4.0 |
| **MathVista** | [AI4Math/MathVista](https://huggingface.co/datasets/AI4Math/MathVista) | 1K | 150 | CC BY-NC 4.0 |

### 🔤 OCR & 图表理解（2 个）

| 数据集 | 来源 | 训练 | 评估 | 许可 |
|--------|------|:----:|:----:|------|
| **OCR-VQA** | [lmms-lab/OCR-VQA](https://huggingface.co/datasets/lmms-lab/OCR-VQA) | 20K | 1K | CC BY 4.0 |
| **ChartQA** | [HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) | 10K | 1K | CC BY 4.0 |

### 🇨🇳 中文理科 & 数学（5 个）

| 数据集 | 来源 | 训练 | 评估 | 许可 |
|--------|------|:----:|:----:|------|
| **C-Eval** | [ceval/ceval-exam](https://huggingface.co/datasets/ceval/ceval-exam) | 2.7K | 500 | CC BY-NC-SA 4.0 |
| **CMMLU** | [haonan-li/cmmlu](https://huggingface.co/datasets/haonan-li/cmmlu) | 11.9K | 1K | CC BY-NC-SA 4.0 |
| **Ape210K** | [Edmond-z/Ape210K-parquet](https://huggingface.co/datasets/Edmond-z/Ape210K-parquet) | 20K | 1K | Apache 2.0 |
| **OpenR1-Math K12** | [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | 20K | 1K | Apache 2.0 |
| **Gaokao MathQA** | [luckypanda/gaokao-math](https://huggingface.co/datasets/luckypanda/gaokao-math) | 351 | 351 | MIT |
| **Gaokao MathCloze** | [luckypanda/gaokao-math](https://huggingface.co/datasets/luckypanda/gaokao-math) | 118 | 118 | MIT |

### 📖 语言理解（1 个）

| 数据集 | 来源 | 训练 | 评估 | 许可 |
|--------|------|:----:|:----:|------|
| **RACE** | [ehovy/race](https://huggingface.co/datasets/ehovy/race) | 10K | 1K | MIT |

**合计**：~222K 训练 / 8,950 评估 | 中文图文占比 52.2%

---

## 🔒 评估集与训练集严格分离

`download_all_data.py` 的 `create_eval_set()` 会从训练集中**真正移除**评估样本（不是简单采样），确保评估有效性。

---

## ⚖️ 许可协议汇总

| 协议 | 数据集 | 使用限制 |
|------|--------|---------|
| **Apache 2.0** | CMMU, Ape210K, OpenR1-Math, windata-math, We-Math | 商用 OK |
| **MIT** | MMSciBench, Gaokao 系列, RACE | 商用 OK |
| **CC BY 4.0** | OCR-VQA, ChartQA | 商用 OK，需署名 |
| **CC BY-NC 4.0** | Geo170K, ScienceQA, MathVerse, MathVista | **禁止商用** |
| **CC BY-NC-SA 4.0** | C-Eval, CMMLU, CMMMU, M3Exam | **禁止商用**，需继承协议 |

> 💡 **使用建议**：
> - 学术研究：所有数据集均可使用
> - 商业项目：避免使用 CC BY-NC-* 数据集，或替换为开源对应物
> - 重新分发：保留原始许可声明

---

## 🛠️ 工具脚本

| 脚本 | 功能 |
|------|------|
| `download_all_data.py` | 一键下载所有数据集，自动分离 eval |
| `scripts/convert_edu_data.py` | 单个数据集转换（22 个 converter） |
| `scripts/convert_windata.py` | windata 多子集转换 |

---

## ❓ 常见问题

**Q1: 下载失败怎么办？**
```bash
# 国内用户使用 ModelScope 镜像
export HF_ENDPOINT=https://hf-mirror.com
python download_all_data.py
```

**Q2: 如何添加新数据集？**
1. 在 `scripts/convert_edu_data.py` 添加 `convert_xxx()` 函数
2. 在 `CONVERTERS` 字典注册
3. 在 `download_all_data.py` 的 `DATASETS_CONFIG` 添加配置
4. 在本 DATA.md 添加说明

**Q3: 训练集和评估集重叠怎么办？**
```bash
# 删除旧的（可能是旧版本下载的）后重新下载
Remove-Item dataset\edu_*.parquet, dataset\eval\eval_*.parquet
python download_all_data.py
```

**Q4: 如何只下载部分数据集？**
```bash
python download_all_data.py --datasets scienceqa ceval cmmu
```

---

## 📚 参考资源

- **基座模型**：[Qwen2-VL](https://github.com/QwenLM/Qwen-VL)
- **架构参考**：[MiniMind-V](https://github.com/jingyaogong/minimind-v)
- **HuggingFace**：[datasets 文档](https://huggingface.co/docs/datasets)

---

## 📝 更新日志

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-06 | v3.0 | 新增 4 个中文多学科数据集（CMMU/CMMMU/M3Exam/MMSciBench）；删除 CMM-Math/MathReal；评估集与训练集严格分离 |
| 2026-06 | v2.0 | 新增 3 个中文核心图文（We-Math 2.0/Geo170K/windata-math）；共 19 个训练数据集 |
| 2025-xx | v1.0 | 初始版本，13 个数据集 |
