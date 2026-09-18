# -*- coding: utf-8 -*-
"""
train_clue_emotion_roberta.py

Fine-tune hfl/chinese-roberta-wwm-ext on CLUE fine-grained Chinese emotion dataset.

Input files (JSONL):
- train.txt
- valid.txt
- test.txt

Each line format:
{"id": 0, "content": "...", "label": "sadness"}

Main features:
- JSONL reading
- class-weighted loss for imbalanced labels
- evaluation on valid and test
- save best model
- export prediction files and reports

Environment:
- Python 3.11
- torch 2.11.0 + CUDA
- transformers 4.37.2
- datasets 2.18.0
- accelerate 0.27.2
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)


# =========================
# 配置区
# =========================

@dataclass
class Config:
    # ---- 文件路径 ----
    train_file: str = "train.txt"
    valid_file: str = "valid.txt"
    test_file: str = "test.txt"

    # ---- 输出路径 ----
    output_dir: str = r".\outputs\clue_emotion_roberta_7cls"

    # ---- 模型 ----
    model_name: str = "hfl/chinese-roberta-wwm-ext"
    max_length: int = 128
    num_labels: int = 7

    # ---- 训练参数 ----
    seed: int = 42
    num_train_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1

    logging_steps: int = 50
    save_total_limit: int = 2
    early_stopping_patience: int = 2

    metric_for_best_model: str = "macro_f1"
    greater_is_better: bool = True

    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 0   # Windows 稳妥设置
    report_to: str = "none"

    # ---- 标签映射 ----
    label2id: Dict[str, int] = None
    id2label: Dict[int, str] = None


CFG = Config()

CFG.label2id = {
    "sadness": 0,
    "happiness": 1,
    "disgust": 2,
    "anger": 3,
    "like": 4,
    "surprise": 5,
    "fear": 6
}
CFG.id2label = {v: k for k, v in CFG.label2id.items()}


# =========================
# 工具函数
# =========================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_jsonl(file_path: str) -> pd.DataFrame:
    """
    Read JSONL file.
    Expected keys per line:
    - id
    - content
    - label
    """
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {file_path}, line {line_idx}: {e}")

    df = pd.DataFrame(rows)
    return df


def validate_dataframe(df: pd.DataFrame, label2id: Dict[str, int], file_name: str):
    required_cols = ["id", "content", "label"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{file_name} is missing required columns: {missing_cols}")

    invalid_labels = sorted(set(df["label"].unique()) - set(label2id.keys()))
    if invalid_labels:
        raise ValueError(f"{file_name} has invalid labels: {invalid_labels}")


def preprocess_dataframe(df: pd.DataFrame, label2id: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()

    df["content"] = df["content"].astype(str).fillna("").str.strip()
    df = df[df["content"] != ""].reset_index(drop=True)

    df["labels"] = df["label"].map(label2id).astype(int)
    return df


def build_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(
        df[["id", "content", "labels"]],
        preserve_index=False
    )


def tokenize_function(examples, tokenizer, max_length: int):
    return tokenizer(
        examples["content"],
        truncation=True,
        max_length=max_length,
        padding=False
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    weighted_f1 = f1_score(labels, preds, average="weighted")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


def compute_class_weights(train_labels: List[int], num_labels: int) -> torch.Tensor:
    """
    Compute balanced class weights from training labels.
    """
    class_indices = np.arange(num_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_indices,
        y=np.array(train_labels)
    )
    return torch.tensor(weights, dtype=torch.float)


# =========================
# 自定义 Trainer：加入 class-weighted loss
# =========================

class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids", None)
        )
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# =========================
# 导出预测结果
# =========================

def export_predictions(
    trainer: Trainer,
    dataset: Dataset,
    original_df: pd.DataFrame,
    output_csv: str,
    report_json: str,
    id2label: Dict[int, str]
):
    pred_output = trainer.predict(dataset)
    logits = pred_output.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_ids = probs.argmax(axis=1)
    pred_scores = probs.max(axis=1)

    pred_labels = [id2label[int(i)] for i in pred_ids]

    out_df = original_df.copy()
    out_df["pred_id"] = pred_ids
    out_df["pred_label"] = pred_labels
    out_df["pred_score"] = pred_scores

    # 展开各类概率
    for i in range(len(id2label)):
        out_df[f"prob_{id2label[i]}"] = probs[:, i]

    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    report = classification_report(
        original_df["labels"].values,
        pred_ids,
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=4,
        output_dict=True
    )
    save_json(report, report_json)


# =========================
# 主程序
# =========================

def main():
    print("=" * 90)
    print("CLUE Emotion 7-class Fine-tuning Started")
    print("=" * 90)

    ensure_dir(CFG.output_dir)
    set_seed(CFG.seed)

    # 1. 读取数据
    print("\n[Step 1] Loading JSONL files...")
    train_df = read_jsonl(CFG.train_file)
    valid_df = read_jsonl(CFG.valid_file)
    test_df = read_jsonl(CFG.test_file)

    validate_dataframe(train_df, CFG.label2id, "train.txt")
    validate_dataframe(valid_df, CFG.label2id, "valid.txt")
    validate_dataframe(test_df, CFG.label2id, "test.txt")

    train_df = preprocess_dataframe(train_df, CFG.label2id)
    valid_df = preprocess_dataframe(valid_df, CFG.label2id)
    test_df = preprocess_dataframe(test_df, CFG.label2id)

    print(f"Train size: {len(train_df):,}")
    print(f"Valid size: {len(valid_df):,}")
    print(f"Test size:  {len(test_df):,}")

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts())

    # 2. class weights
    print("\n[Step 2] Computing class weights...")
    class_weights = compute_class_weights(
        train_labels=train_df["labels"].tolist(),
        num_labels=CFG.num_labels
    )
    print("Class weights:")
    for i, w in enumerate(class_weights.tolist()):
        print(f"  {CFG.id2label[i]}: {w:.4f}")

    # 3. tokenizer & dataset
    print("\n[Step 3] Building tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

    train_ds = build_hf_dataset(train_df)
    valid_ds = build_hf_dataset(valid_df)
    test_ds = build_hf_dataset(test_df)

    train_ds = train_ds.map(
    lambda x: tokenize_function(x, tokenizer, CFG.max_length),
    batched=True,
    remove_columns=["id", "content"],
    desc="Tokenizing train set"
    )
    valid_ds = valid_ds.map(
    lambda x: tokenize_function(x, tokenizer, CFG.max_length),
    batched=True,
    remove_columns=["id", "content"],
    desc="Tokenizing valid set"
    )
    test_ds = test_ds.map(
    lambda x: tokenize_function(x, tokenizer, CFG.max_length),
    batched=True,
    remove_columns=["id", "content"],
    desc="Tokenizing test set"
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. 加载模型
    print("\n[Step 4] Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model_name,
        num_labels=CFG.num_labels,
        label2id=CFG.label2id,
        id2label=CFG.id2label
    )

    # 5. 训练参数
    print("\n[Step 5] Preparing training arguments...")
    training_args = TrainingArguments(
        output_dir=CFG.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=CFG.num_train_epochs,
        learning_rate=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
        warmup_ratio=CFG.warmup_ratio,

        per_device_train_batch_size=CFG.per_device_train_batch_size,
        per_device_eval_batch_size=CFG.per_device_eval_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,

        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=CFG.metric_for_best_model,
        greater_is_better=CFG.greater_is_better,
        save_total_limit=CFG.save_total_limit,

        logging_steps=CFG.logging_steps,
        seed=CFG.seed,

        fp16=CFG.fp16,
        bf16=CFG.bf16,

        dataloader_num_workers=CFG.dataloader_num_workers,
        report_to=CFG.report_to
    )

    # 6. Trainer
    print("\n[Step 6] Initializing weighted trainer...")
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG.early_stopping_patience)]
    )

    # 7. 训练
    print("\n[Step 7] Training...")
    trainer.train()

    # 8. 保存最佳模型
    print("\n[Step 8] Saving best model...")
    trainer.save_model(CFG.output_dir)
    tokenizer.save_pretrained(CFG.output_dir)

    save_json(
        {
            "label2id": CFG.label2id,
            "id2label": {str(k): v for k, v in CFG.id2label.items()}
        },
        os.path.join(CFG.output_dir, "label_mapping.json")
    )

    # 9. valid 评估
    print("\n[Step 9] Evaluating on valid set...")
    valid_metrics = trainer.evaluate(valid_ds)
    print("Valid metrics:")
    for k, v in valid_metrics.items():
        print(f"{k}: {v}")

    # 10. test 评估
    print("\n[Step 10] Evaluating on test set...")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    # 11. 导出预测
    print("\n[Step 11] Exporting predictions...")
    export_predictions(
        trainer=trainer,
        dataset=valid_ds,
        original_df=valid_df,
        output_csv=os.path.join(CFG.output_dir, "dev_predictions.csv"),
        report_json=os.path.join(CFG.output_dir, "classification_report_dev.json"),
        id2label=CFG.id2label
    )

    export_predictions(
        trainer=trainer,
        dataset=test_ds,
        original_df=test_df,
        output_csv=os.path.join(CFG.output_dir, "test_predictions.csv"),
        report_json=os.path.join(CFG.output_dir, "classification_report_test.json"),
        id2label=CFG.id2label
    )

    # 12. 训练摘要
    summary = {
        "model_name": CFG.model_name,
        "train_size": int(len(train_df)),
        "valid_size": int(len(valid_df)),
        "test_size": int(len(test_df)),
        "max_length": CFG.max_length,
        "num_train_epochs": CFG.num_train_epochs,
        "learning_rate": CFG.learning_rate,
        "train_batch_size": CFG.per_device_train_batch_size,
        "eval_batch_size": CFG.per_device_eval_batch_size,
        "class_weights": {
            CFG.id2label[i]: float(class_weights[i].item())
            for i in range(CFG.num_labels)
        },
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics
    }
    save_json(summary, os.path.join(CFG.output_dir, "training_summary.json"))

    print("\n" + "=" * 90)
    print("Training finished successfully.")
    print(f"Best model saved to: {CFG.output_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()