"""Shared input validation and runtime controls for the three NLP scripts."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
LABELS = ("sadness", "happiness", "disgust", "anger", "like", "surprise", "fear")


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def choose_device(requested="auto"):
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
    return "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)


def require_file(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    return path


def prepare_output(path, inputs=()):
    path = Path(path).expanduser().resolve()
    if path in {Path(p).expanduser().resolve() for p in inputs}:
        raise ValueError(f"Output must not overwrite an input: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path, encoding, required_columns):
    path = require_file(path)
    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeError as exc:
        raise ValueError(f"Cannot decode {path} as {encoding}; specify --encoding (utf-8-sig or gb18030).") from exc
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}; check --encoding too.")
    return df


def clean_text(series):
    return series.fillna("").astype(str).str.strip()


def validate_probabilities(probs):
    probs = np.asarray(probs)
    if probs.ndim != 2 or not np.isfinite(probs).all():
        raise ValueError("Model probabilities must be a finite 2D array.")
    if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError("Model probabilities are outside [0, 1] or do not sum to 1.")


def validate_emotion_mapping(label2id, id2label, config=None):
    try:
        label2id = {str(k): int(v) for k, v in label2id.items()}
        id2label = {int(k): str(v) for k, v in id2label.items()}
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Invalid emotion label mapping.") from exc
    if set(label2id) != set(LABELS) or set(id2label) != set(range(7)):
        raise ValueError("Emotion label mapping must contain exactly the seven expected labels and IDs 0..6.")
    if {v: k for k, v in label2id.items()} != id2label:
        raise ValueError("label2id and id2label are inconsistent.")
    if config is not None:
        if config.num_labels != 7 or {int(k): v for k, v in config.id2label.items()} != id2label or {k: int(v) for k, v in config.label2id.items()} != label2id:
            raise ValueError("Saved label mapping disagrees with model config; refusing to relabel predictions.")
    return label2id, id2label


def filter_high_confidence(df, text_column, high_conf_column):
    df = df.copy()
    df[text_column] = clean_text(df[text_column])
    flags = pd.to_numeric(df[high_conf_column], errors="coerce")
    if flags.isna().any() or not flags.isin([0, 1]).all():
        raise ValueError(f"{high_conf_column} must contain only 0 or 1.")
    df = df.loc[(flags == 1) & (df[text_column] != "")].reset_index(drop=True)
    if df.empty:
        raise ValueError("No high-confidence non-empty comments remain; threshold is unchanged.")
    return df


def sample_per_class(df, count, seed):
    if count is None:
        return df
    samples = []
    for label in LABELS:
        group = df.loc[df["label"] == label]
        if len(group) < count:
            raise ValueError(f"Need {count} rows for label {label}, found {len(group)}.")
        samples.append(group.sample(n=count, random_state=seed))
    return pd.concat(samples, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
