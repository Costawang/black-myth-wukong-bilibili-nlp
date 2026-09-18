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

The training script records an environment of Python 3.11, PyTorch 2.11.0 with CUDA, Transformers 4.37.2, Datasets 2.18.0, and Accelerate 0.27.2. The included `requirements.txt` lists dependencies without forcing versions that were not independently verified from an exported environment.

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
├── .gitignore
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

Create an isolated environment and install the declared dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The scripts use paths relative to the process working directory. Review their configuration blocks before execution.

1. `RoBERTa模型训练_FP16.py` trains the seven-class classifier. Point `train_file`, `valid_file`, and `test_file` to the included CLUE split directory, or run it from a directory containing those files.
2. `情感二分类正负推理.py` expects `blackmyth_cleaned_round1.csv`, which is not included, and writes `blackmyth_sentiment_110m_full_output.csv`.
3. `用于情感多分类推理任务的训练后的RoBERTa模型.py` expects the binary output and a trained model under `outputs/clue_emotion_roberta_7cls`.

The archive provides the three scripts, CLUE data splits, and consolidated final CSV. It does not include the raw Bilibili input, intermediate binary result, trained model checkpoint, or evaluation artifacts. Full end-to-end reruns require those missing inputs or regeneration of the model checkpoint.

## Research context

This project sits at the intersection of computational communication research, natural language processing, and social-media analysis. The packaged outputs are suitable for aggregate descriptive analysis, with model uncertainty and duplicate records taken into account.

## Ethical considerations

The data consists of publicly accessible social-media comment text used for research. The consolidated CSV does not include username, user ID, email, IP address, profile, or gender fields. One telephone number embedded in a comment was replaced with `[REDACTED_PHONE]` in this repository copy; the source ZIP remains unchanged. Free-text data can still contain indirect identifiers or quoted personal information, so the dataset should not be treated as fully anonymized. Researchers should minimize redistribution, avoid attempts to identify commenters, and follow applicable platform terms and research-ethics requirements.

## Author

[Costa Wang](https://github.com/Costawang)
