import math
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# 基本配置
# =========================
MODEL_NAME = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"

INPUT_FILE = "blackmyth_cleaned_round1.csv"
OUTPUT_FILE = "blackmyth_sentiment_110m_full_output.csv"

TEXT_COL = "comment_text"

# 全量跑
SAMPLE_SIZE = None

MAX_LENGTH = 512
BATCH_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_GPU_MEMORY = True

# 第一层高置信筛选阈值
HIGH_CONF_THRESHOLD = 0.80


# =========================
# 工具函数
# =========================
def get_confidence_band(max_prob: float) -> str:
    if max_prob >= 0.95:
        return "very_high_core"
    if max_prob >= 0.90:
        return "very_high"
    if max_prob >= 0.80:
        return "high"
    if max_prob >= 0.70:
        return "moderate"
    if max_prob >= 0.60:
        return "low_directional"
    return "ambiguous"


def normalise_label(label: str) -> str:
    return str(label).strip().lower()


def build_label_mapping(id2label: Dict[int, str]) -> Dict[int, str]:
    mapping = {}
    for idx, raw_label in id2label.items():
        lab = normalise_label(raw_label)

        if any(k in lab for k in ["pos", "positive", "好", "满意", "1"]):
            mapping[idx] = "positive"
        elif any(k in lab for k in ["neg", "negative", "差", "不满意", "0"]):
            mapping[idx] = "negative"
        else:
            mapping[idx] = raw_label

    mapped_values = set(mapping.values())
    if "positive" not in mapped_values or "negative" not in mapped_values:
        if len(id2label) == 2:
            sorted_ids = sorted(id2label.keys())
            mapping[sorted_ids[0]] = "negative"
            mapping[sorted_ids[1]] = "positive"

    return mapping


def print_gpu_status(prefix: str = ""):
    if torch.cuda.is_available() and PRINT_GPU_MEMORY:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        name = torch.cuda.get_device_name(0)
        print(
            f"{prefix}GPU: {name} | "
            f"allocated={allocated:.2f} GB | "
            f"reserved={reserved:.2f} GB | "
            f"peak={peak:.2f} GB"
        )


# =========================
# 主逻辑
# =========================
def main():
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print_gpu_status("[Before loading model] ")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    if DEVICE == "cuda":
        print_gpu_status("[After loading model] ")

    config = model.config
    id2label = getattr(config, "id2label", None)
    if not id2label:
        raise ValueError("模型 config 中没有 id2label，无法解析情感标签。")

    print("Original id2label:", id2label)
    label_mapping = build_label_mapping(id2label)
    print("Mapped labels:", label_mapping)

    print("Loading CSV...")
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    required_cols = {"comment_date", "comment_text", "like_count", "reply_count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要字段：{missing}")

    if SAMPLE_SIZE is not None:
        df = df.head(SAMPLE_SIZE).copy()
        print(f"Running on sample size: {len(df)}")
    else:
        print(f"Running on full size: {len(df)}")

    texts = df[TEXT_COL].fillna("").astype(str).tolist()

    pred_labels: List[str] = []
    positive_probs: List[float] = []
    negative_probs: List[float] = []
    max_probs: List[float] = []
    confidence_bands: List[str] = []
    is_high_confidence_list: List[int] = []

    num_batches = math.ceil(len(texts) / BATCH_SIZE)

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Inference"):
            batch_texts = texts[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).detach().cpu()

            for row_probs in probs:
                row_probs = row_probs.tolist()
                idx_prob = {idx: row_probs[idx] for idx in range(len(row_probs))}

                pos_prob = None
                neg_prob = None
                for idx, mapped in label_mapping.items():
                    if mapped == "positive":
                        pos_prob = idx_prob[idx]
                    elif mapped == "negative":
                        neg_prob = idx_prob[idx]

                if pos_prob is None or neg_prob is None:
                    raise ValueError(
                        f"无法从标签映射中识别 positive/negative。"
                        f"id2label={id2label}, mapped={label_mapping}"
                    )

                if pos_prob >= neg_prob:
                    pred_label = "positive"
                    max_prob = pos_prob
                else:
                    pred_label = "negative"
                    max_prob = neg_prob

                pred_labels.append(pred_label)
                positive_probs.append(round(float(pos_prob), 6))
                negative_probs.append(round(float(neg_prob), 6))
                max_probs.append(round(float(max_prob), 6))
                confidence_bands.append(get_confidence_band(float(max_prob)))
                is_high_confidence_list.append(int(float(max_prob) >= HIGH_CONF_THRESHOLD))

    out_df = df.copy()
    out_df["pred_label"] = pred_labels
    out_df["positive_prob"] = positive_probs
    out_df["negative_prob"] = negative_probs
    out_df["max_prob"] = max_probs
    out_df["confidence_band"] = confidence_bands
    out_df["is_high_confidence"] = is_high_confidence_list

    out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nPrediction summary:")
    print(out_df["pred_label"].value_counts(dropna=False))

    print("\nConfidence band summary:")
    print(out_df["confidence_band"].value_counts(dropna=False))

    print(f"\nHigh-confidence threshold: {HIGH_CONF_THRESHOLD}")
    print("High-confidence count:")
    print(out_df["is_high_confidence"].value_counts(dropna=False))

    if DEVICE == "cuda":
        print_gpu_status("[After inference] ")


if __name__ == "__main__":
    main()