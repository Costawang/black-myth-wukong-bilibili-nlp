 # -*- coding: utf-8 -*-
"""
infer_blackmyth_with_clue_emotion.py

Use the fine-tuned CLUE 7-class emotion model to run inference
on the high-confidence Black Myth comment pool.

Input:
- blackmyth_sentiment_110m_full_output.csv

Filter:
- is_high_confidence == 1

Output:
- blackmyth_clue_emotion_highconf_output.csv
- blackmyth_clue_emotion_distribution.csv
- blackmyth_clue_positive_summary.json

Added columns:
- emotion_label
- emotion_score
- all_emotion_scores
- prob_sadness
- prob_happiness
- prob_disgust
- prob_anger
- prob_like
- prob_surprise
- prob_fear
- is_positive_emotion_3cls
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================
# 配置区
# =========================

@dataclass
class Config:
    # ---- 模型目录 ----
    model_dir: str = r".\outputs\clue_emotion_roberta_7cls"

    # ---- 输入输出 ----
    input_csv: str = "blackmyth_sentiment_110m_full_output.csv"
    output_csv: str = "blackmyth_clue_emotion_highconf_output.csv"
    distribution_csv: str = "blackmyth_clue_emotion_distribution.csv"
    summary_json: str = "blackmyth_clue_positive_summary.json"

    # ---- 列名 ----
    text_column: str = "comment_text"
    high_conf_column: str = "is_high_confidence"

    # ---- 推理参数 ----
    max_length: int = 512
    batch_size: int = 96

    # ---- 正向三类定义 ----
    positive_3cls: List[str] = None


CFG = Config()
CFG.positive_3cls = ["like", "happiness", "surprise"]


# =========================
# 工具函数
# =========================

def load_label_mapping(model_dir: str):
    """
    Load label mapping from saved model directory.
    """
    mapping_path = os.path.join(model_dir, "label_mapping.json")

    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        label2id = data["label2id"]
        id2label = {int(k): v for k, v in data["id2label"].items()}
    else:
        # fallback
        label2id = {
            "sadness": 0,
            "happiness": 1,
            "disgust": 2,
            "anger": 3,
            "like": 4,
            "surprise": 5,
            "fear": 6
        }
        id2label = {v: k for k, v in label2id.items()}

    return label2id, id2label


def check_required_columns(df: pd.DataFrame, required_columns: List[str]):
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_text_series(series: pd.Series) -> pd.Series:
    return series.astype(str).fillna("").str.strip()


def predict_batch(texts, tokenizer, model, device, max_length: int):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    pred_ids = probs.argmax(axis=1)
    pred_scores = probs.max(axis=1)

    return pred_ids, pred_scores, probs


def build_distribution_table(df: pd.DataFrame, emotion_col: str) -> pd.DataFrame:
    counts = df[emotion_col].value_counts(dropna=False).reset_index()
    counts.columns = ["emotion_label", "count"]
    counts["ratio"] = counts["count"] / counts["count"].sum()
    return counts


# =========================
# 主程序
# =========================

def main():
    print("=" * 90)
    print("Black Myth CLUE Emotion Inference Started")
    print("=" * 90)

    # 1. 加载模型与 tokenizer
    print("\n[Step 1] Loading model and tokenizer...")
    label2id, id2label = load_label_mapping(CFG.model_dir)

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(CFG.model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Model dir: {CFG.model_dir}")

    # 2. 读取 CSV
    print("\n[Step 2] Loading input CSV...")
    df = pd.read_csv(CFG.input_csv, encoding="gb18030")
    check_required_columns(df, [CFG.text_column, CFG.high_conf_column])

    print(f"Original rows: {len(df):,}")

    # 3. 保留高置信评论
    print("\n[Step 3] Filtering high-confidence rows...")
    df = df.copy()
    df[CFG.text_column] = clean_text_series(df[CFG.text_column])

    highconf_df = df[df[CFG.high_conf_column] == 1].copy()
    highconf_df = highconf_df[highconf_df[CFG.text_column] != ""].reset_index(drop=True)

    print(f"High-confidence rows kept: {len(highconf_df):,}")

    if len(highconf_df) == 0:
        raise ValueError("No high-confidence rows found after filtering.")

    # 4. 批量推理
    print("\n[Step 4] Running batch inference...")
    texts = highconf_df[CFG.text_column].tolist()

    all_pred_ids = []
    all_pred_scores = []
    all_probs = []

    for start_idx in tqdm(range(0, len(texts), CFG.batch_size), desc="Infer"):
        batch_texts = texts[start_idx:start_idx + CFG.batch_size]
        pred_ids, pred_scores, probs = predict_batch(
            batch_texts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=CFG.max_length
        )
        all_pred_ids.extend(pred_ids.tolist())
        all_pred_scores.extend(pred_scores.tolist())
        all_probs.append(probs)

    all_probs = np.vstack(all_probs)

    # 5. 整理结果
    print("\n[Step 5] Formatting outputs...")
    ordered_labels = [id2label[i] for i in range(len(id2label))]

    highconf_df["emotion_id"] = all_pred_ids
    highconf_df["emotion_label"] = [id2label[int(i)] for i in all_pred_ids]
    highconf_df["emotion_score"] = all_pred_scores

    # all_emotion_scores 保存成 JSON 字符串
    all_scores_json = []
    for row in all_probs:
        score_dict = {ordered_labels[i]: float(row[i]) for i in range(len(row))}
        all_scores_json.append(json.dumps(score_dict, ensure_ascii=False))
    highconf_df["all_emotion_scores"] = all_scores_json

    # 展开每类概率列
    for i, label in enumerate(ordered_labels):
        highconf_df[f"prob_{label}"] = all_probs[:, i]

    # 正向三类标记
    highconf_df["is_positive_emotion_3cls"] = highconf_df["emotion_label"].isin(CFG.positive_3cls).astype(int)

    # 6. 保存主结果
    print("\n[Step 6] Saving main output CSV...")
    highconf_df.to_csv(CFG.output_csv, index=False, encoding="utf-8-sig")

    # 7. 七类分布统计
    print("\n[Step 7] Building emotion distribution table...")
    dist_df = build_distribution_table(highconf_df, "emotion_label")
    dist_df.to_csv(CFG.distribution_csv, index=False, encoding="utf-8-sig")

    # 8. 正向三类统计
    print("\n[Step 8] Building positive-3-class summary...")
    total_rows = len(highconf_df)
    positive_rows = int(highconf_df["is_positive_emotion_3cls"].sum())
    positive_ratio = positive_rows / total_rows

    positive_breakdown = (
        highconf_df[highconf_df["emotion_label"].isin(CFG.positive_3cls)]["emotion_label"]
        .value_counts(dropna=False)
        .to_dict()
    )

    full_distribution = highconf_df["emotion_label"].value_counts(dropna=False).to_dict()

    summary = {
        "input_csv": CFG.input_csv,
        "output_csv": CFG.output_csv,
        "model_dir": CFG.model_dir,
        "total_high_confidence_comments": total_rows,
        "positive_3cls_definition": CFG.positive_3cls,
        "positive_3cls_count": positive_rows,
        "positive_3cls_ratio": positive_ratio,
        "positive_3cls_is_majority": bool(positive_ratio > 0.5),
        "positive_3cls_breakdown": positive_breakdown,
        "full_emotion_distribution": full_distribution
    }

    with open(CFG.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 9. 打印结果
    print("\n" + "=" * 90)
    print("Inference finished successfully.")
    print(f"Main output saved to: {CFG.output_csv}")
    print(f"Distribution table saved to: {CFG.distribution_csv}")
    print(f"Summary saved to: {CFG.summary_json}")
    print("\nEmotion distribution:")
    print(dist_df)
    print("\nPositive 3-class summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 90)


if __name__ == "__main__":
    main()