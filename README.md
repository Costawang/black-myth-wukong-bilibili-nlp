# Black Myth: Wukong — Bilibili Comment NLP Analysis

This repository contains an NLP workflow and consolidated results for a computational communication study of Bilibili comments about *Black Myth: Wukong*. The code covers binary sentiment inference, seven-class emotion model training, and emotion inference on high-confidence comments.

## Research objective

The project examines how audience responses to *Black Myth: Wukong* are distributed across positive and negative sentiment and seven emotion categories. It supports quantitative analysis of public social-media discourse rather than individual-level profiling.

## Dataset

### Bilibili analysis output

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

### CLUE emotion data

The included JSONL splits use the fields `id`, `content`, and `label`:

| Split | Rows |
| --- | ---: |
| Train | 31,728 |
| Validation | 3,966 |
| Test | 3,967 |

The seven labels are `sadness`, `happiness`, `disgust`, `anger`, `like`, `surprise`, and `fear`. The training script validates these labels and removes blank content before model training.

## NLP pipeline

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

## Models and methods

- Binary sentiment model: `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`
- Seven-class base model: `hfl/chinese-roberta-wwm-ext`
- Frameworks: PyTorch, Hugging Face Transformers, Datasets, pandas, NumPy, and scikit-learn
- Training controls: class-weighted cross-entropy, early stopping, fixed random seed, and FP16 GPU training
- Evaluation implemented in code: accuracy, macro F1, weighted F1, and per-class classification reports

The runtime was checked on Windows with Python 3.11.9, PyTorch 2.11.0+cu128, Transformers 4.37.2, Datasets 2.18.0, and Accelerate 0.27.2. `requirements.txt` pins the direct dependencies; `requirements-lock-windows-cu128.txt` records the installed transitive dependency closure for the tested Windows/CUDA environment. See the [runtime validation report](docs/runtime-validation-2026-10-06.md) for coverage and limitations.

## Key packaged results

### Binary sentiment

| Label | Count | Share |
| --- | ---: | ---: |
| Positive | 38,094 | 63.70% |
| Negative | 21,708 | 36.30% |

### Seven-class emotion distribution

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

## Repository structure

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

## Reproducibility and usage

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

### Train and save a seven-class model

The default input paths point to the bundled CLUE splits. This command performs a **five-step smoke test**, not a full research training run:

```powershell
python "RoBERTa模型训练_FP16.py" --output-dir runs/manual-smoke-model --train-per-class 16 --eval-per-class 4 --max-steps 5 --batch-size 8 --eval-batch-size 8 --device cuda --seed 42
```

The output directory must be new or empty. It receives model/tokenizer files, label mappings, validation/test predictions and classification reports, and `training_summary.json`. The smoke sample contains 112 training rows and 28 rows in each evaluation split. Its metrics are not evidence of model quality.

For full training, omit `--train-per-class`, `--eval-per-class`, and `--max-steps`; the existing five-epoch, class-weighted training recipe remains the default. Use `--train-file`, `--valid-file`, and `--test-file` to override the bundled data, and `--model` to select a cached model directory or Hugging Face model ID. `--device cpu` disables FP16; `--precision fp32` is also available. Full retraining was not part of this validation.

### Run the two-stage inference chain

Supply a local cleaned CSV containing `comment_date`, `comment_text`, `like_count`, and `reply_count`. The original cleaned input and trained weights are **not distributed in this repository**. The packaged final CSV is historical output, not an independent raw input for reproducing the study.

```powershell
python "情感二分类正负推理.py" --input "D:\data\blackmyth_cleaned_round1.csv" --output runs/manual/binary.csv --sample-size 200 --seed 42 --batch-size 8 --device cuda
python "用于情感多分类推理任务的训练后的RoBERTa模型.py" --input runs/manual/binary.csv --model "D:\models\clue_emotion_roberta_7cls" --output-dir runs/manual/emotion --batch-size 8 --device cuda
```

Use a trained model you already have, or `runs/manual-smoke-model` to check the short-trained model's save/load path. The emotion command writes `blackmyth_clue_emotion_highconf_output.csv`, `blackmyth_clue_emotion_distribution.csv`, and `blackmyth_clue_positive_summary.json`. The confidence threshold stays at **0.80**, and positive emotions remain `like`, `happiness`, and `surprise`.

New CSV outputs and default inputs consistently use **UTF-8-SIG**. Add `--encoding gb18030` only when reading a known legacy GB18030 file; do not use it for newly generated binary output. Empty/missing inputs, invalid flags, conflicting model label mappings and a pool with no high-confidence non-empty comments produce explicit errors. Unknown binary label semantics are rejected instead of guessing their order. Inference writes its requested output paths, so choose a separate output directory for each experiment.

### Automated regression and real-model smoke checks

```powershell
# Synthetic regression tests: no model download or private data needed.
python -m unittest discover -s tests -v

# Real GPU training and GPU/CPU inference; requires local input and a trained model.
python scripts/smoke_test.py --input "D:\data\blackmyth_cleaned_round1.csv" --existing-model "D:\models\clue_emotion_roberta_7cls" --run-dir runs/smoke-check --local-files-only
```

`--run-dir` must not already exist. The runner samples 200 comments with seed 42 using only the four original input fields, performs five training steps, checks both new and existing emotion models, runs two CPU comments and one synthetic long-text example, and validates probabilities, labels, row alignment and summaries. It runs scripts from a different working directory to test path handling. GPU batch size starts at 8 and retries 4, 2, then 1 only on out-of-memory errors. Other failures stop with a saved log and failure status.

The runner writes local logs and `run_report.json`, including exact commands, versions, code fingerprints and elapsed times. These local files contain paths and sample data and are ignored by Git; publish only a sanitized summary. Models, checkpoints and caches are also ignored. `--local-files-only` requires cached base/binary models; omit it to allow downloading those public models, or specify local directories with `--base-model` and `--binary-model`. No comments are sent to a remote inference service.

The 2026-10-06 check passed the full small-sample chain, but did not rerun all comments or reproduce the historical research metrics. See the [validation report](docs/runtime-validation-2026-10-06.md).

## Research context

This project sits at the intersection of computational communication research, natural language processing, and social-media analysis. The packaged outputs are suitable for aggregate descriptive analysis, with model uncertainty and duplicate records taken into account.

## Ethical considerations

The data consists of publicly accessible social-media comment text used for research. The consolidated CSV does not include username, user ID, email, IP address, profile, or gender fields. One telephone number embedded in a comment was replaced with `[REDACTED_PHONE]` in this repository copy; the source ZIP remains unchanged. Free-text data can still contain indirect identifiers or quoted personal information, so the dataset should not be treated as fully anonymized. Researchers should minimize redistribution, avoid attempts to identify commenters, and follow applicable platform terms and research-ethics requirements.

## Author

[Costa Wang](https://github.com/Costawang)
