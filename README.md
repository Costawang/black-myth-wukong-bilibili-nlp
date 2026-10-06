<a id="top"></a>

# 黑神话：悟空 — B站评论 NLP 分析

Black Myth: Wukong — Bilibili Comment NLP Analysis

[中文](#中文说明) | [English](#english)

## 中文说明

本仓库提供面向《黑神话：悟空》B站评论计算传播学研究的自然语言处理（NLP）流程与汇总结果。代码涵盖情感正负二分类推理、七分类情绪模型训练，以及针对高置信评论的七分类情绪推理。

### 研究目标

分析受众对《黑神话：悟空》的评论在正负情感及七种情绪类别中的分布，为公开社交媒体话语的量化研究提供支持。研究关注群体层面的表达特征，不进行个人画像分析。

### 数据集

#### B站评论分析结果

`文本情绪二分类+多分类全量结果.csv` 是使用 **GB18030** 编码的历史汇总结果文件，包含：

- 59,802 行、22 列。
- 评论日期范围：2024-08-08 至 2026-03-11。
- 打包文件所含字段无缺失值。
- 情感二分类标签、各类别概率、最大预测概率和置信度分档。
- 七分类情绪标签、各类别概率，以及正向三类情绪标记。

主要字段如下：

| 字段类别 | 列名 |
| --- | --- |
| 评论数据 | `comment_date`、`comment_text`、`like_count`、`reply_count` |
| 情感二分类 | `pred_label`、`positive_prob`、`negative_prob`、`max_prob`、`confidence_band`、`is_high_confidence` |
| 情绪分类结果 | `emotion_id`、`emotion_label`、`emotion_score`、`all_emotion_scores` |
| 各类别概率 | `prob_sadness`、`prob_happiness`、`prob_disgust`、`prob_anger`、`prob_like`、`prob_surprise`、`prob_fear` |
| 衍生指标 | `is_positive_emotion_3cls` |

仓库内结果的所有行均满足 `is_high_confidence = 1`，与二分类脚本的 `max_prob >= 0.80` 筛选阈值一致。原始压缩包未包含采集原始文件或上游清洗脚本，因此无法仅凭仓库内容独立核实采集方法和前期清洗过程。汇总文件中有 1,588 行完全重复记录，另有 7,348 行的 `comment_text` 与更早的某行相同；本项目不宣称该文件已经完成去重。

#### CLUE 情绪数据

仓库提供 JSONL 格式的训练、验证和测试划分，使用 `id`、`content` 和 `label` 三个字段：

| 数据划分 | 行数 |
| --- | ---: |
| 训练集 | 31,728 |
| 验证集 | 3,966 |
| 测试集 | 3,967 |

七种标签分别为 `sadness`（悲伤）、`happiness`（快乐）、`disgust`（厌恶）、`anger`（愤怒）、`like`（喜爱）、`surprise`（惊讶）和 `fear`（恐惧）。训练脚本会校验标签，并在训练前移除空白文本。

### NLP 处理流程

```text
B站评论输入
  -> 情感正负二分类推理
  -> 置信度评分与筛选（>= 0.80）
  -> 七分类情绪推理
  -> 各类别概率与汇总指标
  -> 用于量化分析的汇总 CSV
```

七分类情绪模型另有独立的训练流程：

```text
CLUE JSONL 数据划分
  -> 文本规范化与标签映射
  -> 分词与编码
  -> 使用类别加权损失微调 RoBERTa
  -> 验证集与测试集评估
  -> 模型检查点与预测报告
```

### 模型与方法

- 情感二分类模型：`IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`。
- 七分类基础模型：`hfl/chinese-roberta-wwm-ext`。
- 主要框架：PyTorch、Hugging Face Transformers、Datasets、pandas、NumPy 和 scikit-learn。
- 训练设置：类别加权交叉熵损失、早停、固定随机种子，以及 GPU FP16 训练。
- 代码实现的评估指标：准确率、宏平均 F1、加权 F1 和逐类别分类报告。

已在 Windows 环境下使用 Python 3.11.9、PyTorch 2.11.0+cu128、Transformers 4.37.2、Datasets 2.18.0 和 Accelerate 0.27.2 完成运行验证。`requirements.txt` 固定直接依赖版本；`requirements-lock-windows-cu128.txt` 记录已验证 Windows/CUDA 环境中安装的直接与间接依赖版本。验证覆盖范围与限制见[运行验证报告](docs/runtime-validation-2026-10-06.md)。

### 仓库内历史结果

#### 情感二分类

| 标签 | 数量 | 占比 |
| --- | ---: | ---: |
| 正向（Positive） | 38,094 | 63.70% |
| 负向（Negative） | 21,708 | 36.30% |

#### 七分类情绪分布

| 情绪 | 数量 | 占比 |
| --- | ---: | ---: |
| 喜爱（Like） | 13,176 | 22.03% |
| 快乐（Happiness） | 11,945 | 19.97% |
| 愤怒（Anger） | 11,403 | 19.07% |
| 悲伤（Sadness） | 8,885 | 14.86% |
| 厌恶（Disgust） | 8,405 | 14.05% |
| 惊讶（Surprise） | 4,678 | 7.82% |
| 恐惧（Fear） | 1,310 | 2.19% |

脚本将 `like`、`happiness` 和 `surprise` 定义为正向情绪。三类合计 29,799 条评论，占仓库内高置信样本的 49.83%。这些标签来自模型预测，并非人工核验的真实标签。

原始压缩包未包含保存的训练摘要或分类报告，因此本仓库未报告基于原始完整训练的留出集性能指标。

### 仓库结构

```text
.
├── README.md
├── requirements.txt
├── requirements-lock-windows-cu128.txt
├── .gitignore
├── pipeline_runtime.py
├── scripts/smoke_test.py
├── tests/test_pipeline.py
├── docs/runtime-validation-2026-10-06.md
├── RoBERTa模型训练_FP16.py
├── 情感二分类正负推理.py
├── 用于情感多分类推理任务的训练后的RoBERTa模型.py
├── 文本情绪二分类+多分类全量结果.csv
└── CLUE文本数据集（情感多分类任务）_GitHub开源数据集/
    ├── train.txt
    ├── valid.txt
    └── test.txt
```

### 复现与使用方法

在仓库根目录打开 PowerShell，使用以下命令准备已验证的 Windows/CUDA 依赖版本：

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-lock-windows-cu128.txt
python -m pip check
```

锁文件记录的是已通过测试的现有环境；此次验证未重新安装一个全新环境。同一个支持 CUDA 的 PyTorch 安装包也可以在 CPU 上运行。对于其他平台，`requirements.txt` 仅固定直接依赖版本；其他平台及替代依赖版本尚未经过验证。

三份脚本均支持 `--help`。默认路径以仓库目录为基准，不受当前工作目录影响；命令行显式传入的相对路径以启动命令时的工作目录为基准。外部数据和模型建议使用绝对路径。选择输入文件或批大小时无需修改源码。

#### 训练并保存七分类模型

默认输入路径指向仓库内的 CLUE 数据划分。以下命令执行的是 **5 步小样本运行验证（smoke test）**，并非完整研究训练：

```powershell
python "RoBERTa模型训练_FP16.py" --output-dir runs/manual-smoke-model --train-per-class 16 --eval-per-class 4 --max-steps 5 --batch-size 8 --eval-batch-size 8 --device cuda --seed 42
```

输出目录必须不存在或为空。脚本会保存模型与分词器、标签映射、验证集和测试集预测、分类报告，以及 `training_summary.json`。小样本训练集包含 112 行，验证集和测试集各包含 28 行；这些小样本指标不能作为模型质量的证据。

如需完整训练，省略 `--train-per-class`、`--eval-per-class` 和 `--max-steps`，默认仍使用原有的 5 轮类别加权训练方案。可通过 `--train-file`、`--valid-file`、`--test-file` 指定其他数据，通过 `--model` 指定缓存模型目录或 Hugging Face 模型 ID。`--device cpu` 会关闭 FP16，也可显式使用 `--precision fp32`。此次验证未进行完整重训。

#### 运行两级推理链路

需要提供本地清洗后的 CSV，包含 `comment_date`、`comment_text`、`like_count` 和 `reply_count` 四列。**本仓库不分发原始清洗输入和已训练模型权重。** 仓库附带的最终 CSV 是历史分析输出，不能视为用于独立复现研究的原始输入。

```powershell
python "情感二分类正负推理.py" --input "D:\data\blackmyth_cleaned_round1.csv" --output runs/manual/binary.csv --sample-size 200 --seed 42 --batch-size 8 --device cuda
python "用于情感多分类推理任务的训练后的RoBERTa模型.py" --input runs/manual/binary.csv --model "D:\models\clue_emotion_roberta_7cls" --output-dir runs/manual/emotion --batch-size 8 --device cuda
```

七分类推理可使用已有训练模型，也可指定 `runs/manual-smoke-model` 验证短训模型的保存与加载链路。情绪推理命令会输出 `blackmyth_clue_emotion_highconf_output.csv`、`blackmyth_clue_emotion_distribution.csv` 和 `blackmyth_clue_positive_summary.json`。高置信筛选阈值保持为 **0.80**，正向情绪仍定义为 `like`、`happiness` 和 `surprise`。

新生成的 CSV 和默认输入统一使用 **UTF-8-SIG** 编码。只有读取已知的历史 GB18030 文件时，才添加 `--encoding gb18030`；新生成的二分类结果不应使用该参数。遇到输入为空或缺失、置信标记非法、模型标签映射冲突，或筛选后没有高置信非空评论时，脚本会明确报错。二分类标签语义无法确定时会拒绝继续，而不会猜测标签顺序。推理会写入指定输出路径，请为每次实验选择独立的输出目录。

#### 自动回归测试与真实模型运行验证

```powershell
# 使用合成数据进行回归测试，无需下载模型或提供私人数据。
python -m unittest discover -s tests -v

# 真实 GPU 训练及 GPU/CPU 推理，需要本地输入和已训练模型。
python scripts/smoke_test.py --input "D:\data\blackmyth_cleaned_round1.csv" --existing-model "D:\models\clue_emotion_roberta_7cls" --run-dir runs/smoke-check --local-files-only
```

`--run-dir` 指定的目录必须尚不存在。验证脚本使用随机种子 42 抽取 200 条评论，仅保留四个原始输入字段；执行 5 步训练，分别检查新训练模型和已有情绪模型，并运行两条 CPU 样本和一条合成超长文本，同时校验概率、标签、行对应关系及汇总结果。脚本会从不同工作目录启动各阶段，以验证路径处理。GPU 批大小从 8 开始，仅在显存不足时依次尝试 4、2、1；其他错误会停止运行，并保存日志和失败状态。

验证脚本在本地保存日志及 `run_report.json`，记录完整命令、依赖版本、代码指纹和耗时。这些文件包含路径和样本数据，已由 Git 忽略；公开发布时应仅提供脱敏摘要。模型、检查点和缓存同样被忽略。`--local-files-only` 要求基础模型和二分类模型已缓存；省略该参数即可允许下载这些公开模型，也可通过 `--base-model` 和 `--binary-model` 指定本地目录。评论不会被发送到远程推理服务。

2026-10-06 的检查已通过完整的小样本链路，但未重新推理全部评论，也未复现历史研究指标。详情见[运行验证报告](docs/runtime-validation-2026-10-06.md)。

### 研究背景

本项目结合计算传播学、自然语言处理与社交媒体分析。仓库内的汇总结果可用于群体层面的描述性分析，使用时应考虑模型预测的不确定性和重复记录的影响。

### 研究伦理与隐私

数据来自用于研究的公开社交媒体评论。汇总 CSV 不包含用户名、用户 ID、电子邮箱、IP 地址、个人资料或性别字段。仓库副本已将一条评论中出现的电话号码替换为 `[REDACTED_PHONE]`，原始 ZIP 未被修改。自由文本仍可能包含间接身份线索或被引用的个人信息，因此不能将该数据集视为已经完全匿名化。研究者应尽量减少再分发，避免识别评论者身份，并遵循适用的平台规则和研究伦理要求。

### 作者

[Costa Wang](https://github.com/Costawang)

[返回顶部 / Back to top](#top)

---

## English

This repository contains an NLP workflow and consolidated results for a computational communication study of Bilibili comments about *Black Myth: Wukong*. The code covers binary sentiment inference, seven-class emotion model training, and emotion inference on high-confidence comments.

### Research objective

The project examines how audience responses to *Black Myth: Wukong* are distributed across positive and negative sentiment and seven emotion categories. It supports quantitative analysis of public social-media discourse rather than individual-level profiling.

### Dataset

#### Bilibili analysis output

`文本情绪二分类+多分类全量结果.csv` is a consolidated, GB18030-encoded result file with:

- 59,802 rows and 22 columns
- comment dates from 2024-08-08 to 2026-03-11
- no missing values in the packaged columns
- binary sentiment labels, class probabilities, maximum probability, and confidence bands
- seven-class emotion labels, per-class probabilities, and a three-class positive-emotion indicator

The main fields are:

| Field group | Columns |
| --- | --- |
| Comment data | `comment_date`, `comment_text`, `like_count`, `reply_count` |
| Binary sentiment | `pred_label`, `positive_prob`, `negative_prob`, `max_prob`, `confidence_band`, `is_high_confidence` |
| Emotion output | `emotion_id`, `emotion_label`, `emotion_score`, `all_emotion_scores` |
| Per-class probabilities | `prob_sadness`, `prob_happiness`, `prob_disgust`, `prob_anger`, `prob_like`, `prob_surprise`, `prob_fear` |
| Derived indicator | `is_positive_emotion_3cls` |

All packaged rows have `is_high_confidence = 1`, consistent with the binary script's threshold of `max_prob >= 0.80`. The archive does not contain the raw collection file or the upstream cleaning script, so the collection method and earlier cleaning stages cannot be independently verified here. The consolidated file contains 1,588 exact duplicate rows, and 7,348 rows repeat an earlier `comment_text`; no deduplication claim is made.

#### CLUE emotion data

The included JSONL splits use the fields `id`, `content`, and `label`:

| Split | Rows |
| --- | ---: |
| Train | 31,728 |
| Validation | 3,966 |
| Test | 3,967 |

The seven labels are `sadness`, `happiness`, `disgust`, `anger`, `like`, `surprise`, and `fear`. The training script validates these labels and removes blank content before model training.

### NLP pipeline

```text
Bilibili comment input
  -> binary sentiment inference
  -> confidence scoring and thresholding (>= 0.80)
  -> seven-class emotion inference
  -> per-class probabilities and summary indicators
  -> consolidated CSV for quantitative analysis
```

The repository also includes the separate training path for the seven-class emotion model:

```text
CLUE JSONL splits
  -> text normalization and label mapping
  -> tokenization
  -> class-weighted RoBERTa fine-tuning
  -> validation/test evaluation
  -> model checkpoint and prediction reports
```

### Models and methods

- Binary sentiment model: `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`
- Seven-class base model: `hfl/chinese-roberta-wwm-ext`
- Frameworks: PyTorch, Hugging Face Transformers, Datasets, pandas, NumPy, and scikit-learn
- Training controls: class-weighted cross-entropy, early stopping, fixed random seed, and FP16 GPU training
- Evaluation implemented in code: accuracy, macro F1, weighted F1, and per-class classification reports

The runtime was checked on Windows with Python 3.11.9, PyTorch 2.11.0+cu128, Transformers 4.37.2, Datasets 2.18.0, and Accelerate 0.27.2. `requirements.txt` pins the direct dependencies; `requirements-lock-windows-cu128.txt` records the installed transitive dependency closure for the tested Windows/CUDA environment. See the [runtime validation report](docs/runtime-validation-2026-10-06.md) for coverage and limitations.

### Key packaged results

#### Binary sentiment

| Label | Count | Share |
| --- | ---: | ---: |
| Positive | 38,094 | 63.70% |
| Negative | 21,708 | 36.30% |

#### Seven-class emotion distribution

| Emotion | Count | Share |
| --- | ---: | ---: |
| Like | 13,176 | 22.03% |
| Happiness | 11,945 | 19.97% |
| Anger | 11,403 | 19.07% |
| Sadness | 8,885 | 14.86% |
| Disgust | 8,405 | 14.05% |
| Surprise | 4,678 | 7.82% |
| Fear | 1,310 | 2.19% |

The script defines `like`, `happiness`, and `surprise` as positive emotions. Together they account for 29,799 comments, or 49.83% of the packaged high-confidence set. These are model outputs, not manually validated ground-truth labels.

No saved training summary or classification report was included in the archive, so this repository does not report held-out performance metrics.

### Repository structure

```text
.
├── README.md
├── requirements.txt
├── requirements-lock-windows-cu128.txt
├── .gitignore
├── pipeline_runtime.py
├── scripts/smoke_test.py
├── tests/test_pipeline.py
├── docs/runtime-validation-2026-10-06.md
├── RoBERTa模型训练_FP16.py
├── 情感二分类正负推理.py
├── 用于情感多分类推理任务的训练后的RoBERTa模型.py
├── 文本情绪二分类+多分类全量结果.csv
└── CLUE文本数据集（情感多分类任务）_GitHub开源数据集/
    ├── train.txt
    ├── valid.txt
    └── test.txt
```

### Reproducibility and usage

Run the following from the repository root in PowerShell to prepare the tested Windows/CUDA package versions:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-lock-windows-cu128.txt
python -m pip check
```

This lock describes an existing environment that passed the tests; a clean installation was not performed during validation. CPU execution is supported by the same CUDA-enabled wheel. For other platforms, `requirements.txt` pins direct dependencies only; those platforms and alternative dependency versions have not been validated.

All three scripts accept `--help`. Default paths are anchored to the repository, independent of the current working directory. Explicit relative CLI paths are resolved against the caller's working directory; use absolute paths for external data and models. No source edits are needed to select inputs or batch sizes.

#### Train and save a seven-class model

The default input paths point to the bundled CLUE splits. This command performs a **five-step smoke test**, not a full research training run:

```powershell
python "RoBERTa模型训练_FP16.py" --output-dir runs/manual-smoke-model --train-per-class 16 --eval-per-class 4 --max-steps 5 --batch-size 8 --eval-batch-size 8 --device cuda --seed 42
```

The output directory must be new or empty. It receives model/tokenizer files, label mappings, validation/test predictions and classification reports, and `training_summary.json`. The smoke sample contains 112 training rows and 28 rows in each evaluation split. Its metrics are not evidence of model quality.

For full training, omit `--train-per-class`, `--eval-per-class`, and `--max-steps`; the existing five-epoch, class-weighted training recipe remains the default. Use `--train-file`, `--valid-file`, and `--test-file` to override the bundled data, and `--model` to select a cached model directory or Hugging Face model ID. `--device cpu` disables FP16; `--precision fp32` is also available. Full retraining was not part of this validation.

#### Run the two-stage inference chain

Supply a local cleaned CSV containing `comment_date`, `comment_text`, `like_count`, and `reply_count`. The original cleaned input and trained weights are **not distributed in this repository**. The packaged final CSV is historical output, not an independent raw input for reproducing the study.

```powershell
python "情感二分类正负推理.py" --input "D:\data\blackmyth_cleaned_round1.csv" --output runs/manual/binary.csv --sample-size 200 --seed 42 --batch-size 8 --device cuda
python "用于情感多分类推理任务的训练后的RoBERTa模型.py" --input runs/manual/binary.csv --model "D:\models\clue_emotion_roberta_7cls" --output-dir runs/manual/emotion --batch-size 8 --device cuda
```

Use a trained model you already have, or `runs/manual-smoke-model` to check the short-trained model's save/load path. The emotion command writes `blackmyth_clue_emotion_highconf_output.csv`, `blackmyth_clue_emotion_distribution.csv`, and `blackmyth_clue_positive_summary.json`. The confidence threshold stays at **0.80**, and positive emotions remain `like`, `happiness`, and `surprise`.

New CSV outputs and default inputs consistently use **UTF-8-SIG**. Add `--encoding gb18030` only when reading a known legacy GB18030 file; do not use it for newly generated binary output. Empty/missing inputs, invalid flags, conflicting model label mappings and a pool with no high-confidence non-empty comments produce explicit errors. Unknown binary label semantics are rejected instead of guessing their order. Inference writes its requested output paths, so choose a separate output directory for each experiment.

#### Automated regression and real-model smoke checks

```powershell
# Synthetic regression tests: no model download or private data needed.
python -m unittest discover -s tests -v

# Real GPU training and GPU/CPU inference; requires local input and a trained model.
python scripts/smoke_test.py --input "D:\data\blackmyth_cleaned_round1.csv" --existing-model "D:\models\clue_emotion_roberta_7cls" --run-dir runs/smoke-check --local-files-only
```

`--run-dir` must not already exist. The runner samples 200 comments with seed 42 using only the four original input fields, performs five training steps, checks both new and existing emotion models, runs two CPU comments and one synthetic long-text example, and validates probabilities, labels, row alignment and summaries. It runs scripts from a different working directory to test path handling. GPU batch size starts at 8 and retries 4, 2, then 1 only on out-of-memory errors. Other failures stop with a saved log and failure status.

The runner writes local logs and `run_report.json`, including exact commands, versions, code fingerprints and elapsed times. These local files contain paths and sample data and are ignored by Git; publish only a sanitized summary. Models, checkpoints and caches are also ignored. `--local-files-only` requires cached base/binary models; omit it to allow downloading those public models, or specify local directories with `--base-model` and `--binary-model`. No comments are sent to a remote inference service.

The 2026-10-06 check passed the full small-sample chain, but did not rerun all comments or reproduce the historical research metrics. See the [validation report](docs/runtime-validation-2026-10-06.md).

### Research context

This project sits at the intersection of computational communication research, natural language processing, and social-media analysis. The packaged outputs are suitable for aggregate descriptive analysis, with model uncertainty and duplicate records taken into account.

### Ethical considerations

The data consists of publicly accessible social-media comment text used for research. The consolidated CSV does not include username, user ID, email, IP address, profile, or gender fields. One telephone number embedded in a comment was replaced with `[REDACTED_PHONE]` in this repository copy; the source ZIP remains unchanged. Free-text data can still contain indirect identifiers or quoted personal information, so the dataset should not be treated as fully anonymized. Researchers should minimize redistribution, avoid attempts to identify commenters, and follow applicable platform terms and research-ethics requirements.

### Author

[Costa Wang](https://github.com/Costawang)

[Back to top / 返回顶部](#top)
