import argparse
import math
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pipeline_runtime import ROOT, choose_device, positive_int, prepare_output, read_csv, validate_probabilities

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
    aliases = {
        "negative": {"negative", "neg", "差", "不满意", "负面"},
        "positive": {"positive", "pos", "好", "满意", "正面"},
    }
    mapping = {int(idx): next((name for name, labels in aliases.items() if normalise_label(raw) in labels), None)
               for idx, raw in id2label.items()}
    if set(mapping) != {0, 1} or set(mapping.values()) != {"negative", "positive"}:
        raise ValueError(f"Ambiguous binary label mapping: {id2label}; explicit positive/negative model labels are required.")
    return mapping


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Binary sentiment inference; UTF-8-SIG output.")
    parser.add_argument("--input", default=str(ROOT / INPUT_FILE))
    parser.add_argument("--output", default=str(ROOT / "runs" / OUTPUT_FILE))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=positive_int, default=MAX_LENGTH)
    parser.add_argument("--sample-size", type=positive_int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


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
def main(argv=None):
    args = parse_args(argv)
    DEVICE = choose_device(args.device)
    BATCH_SIZE = args.batch_size
    required_cols = {"comment_date", "comment_text", "like_count", "reply_count"}
    df = read_csv(args.input, args.encoding, required_cols)
    if args.sample_size is not None:
        df = df.sample(n=min(args.sample_size, len(df)), random_state=args.seed).reset_index(drop=True)
    if df.empty:
        raise ValueError("Input CSV contains no comments.")
    output_file = prepare_output(args.output, [args.input])
    print(f"Using device: {DEVICE}")
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print_gpu_status("[Before loading model] ")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=args.local_files_only)
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("--max-length exceeds model positional capacity.")
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

    print(f"Running on {len(df)} comments; batch size {BATCH_SIZE}")

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
                max_length=args.max_length,
                return_tensors="pt",
            )

            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).detach().cpu()
            validate_probabilities(probs.numpy())

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

    out_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved to: {output_file}")

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
