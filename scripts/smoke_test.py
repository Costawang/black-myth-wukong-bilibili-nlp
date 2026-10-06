"""Real-model, isolated smoke run. Generated inputs/models/logs stay in runs/."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_runtime import LABELS, clean_text, read_csv, validate_probabilities

BASE_COLUMNS = ["comment_date", "comment_text", "like_count", "reply_count"]
BINARY = "情感二分类正负推理.py"
TRAINING = "RoBERTa模型训练_FP16.py"
EMOTION = "用于情感多分类推理任务的训练后的RoBERTa模型.py"
EMOTION_CSV = "blackmyth_clue_emotion_highconf_output.csv"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_emotion(input_csv, directory):
    binary = pd.read_csv(input_csv, encoding="utf-8-sig")
    expected = binary.loc[(binary.is_high_confidence == 1) & (clean_text(binary.comment_text) != "")].copy()
    output = pd.read_csv(directory / EMOTION_CSV, encoding="utf-8-sig")
    assert len(output) == len(expected) > 0
    assert output.comment_text.tolist() == clean_text(expected.comment_text).tolist()
    validate_probabilities(output[[f"prob_{label}" for label in LABELS]].values)
    assert set(output.emotion_label) <= set(LABELS)
    mapping = dict(enumerate(LABELS))
    assert output.emotion_label.tolist() == output.emotion_id.map(mapping).tolist()
    probabilities = output[[f"prob_{label}" for label in LABELS]].values
    assert np.array_equal(output.emotion_id.values, probabilities.argmax(axis=1))
    assert np.allclose(output.emotion_score.values, probabilities.max(axis=1), atol=1e-5, rtol=0)
    for row, values in zip(output.all_emotion_scores, probabilities):
        scores = json.loads(row)
        assert np.allclose([scores[label] for label in LABELS], values, atol=1e-5, rtol=0)
    positive = output.emotion_label.isin(["like", "happiness", "surprise"]).astype(int)
    assert positive.tolist() == output.is_positive_emotion_3cls.tolist()
    distribution = pd.read_csv(directory / "blackmyth_clue_emotion_distribution.csv", encoding="utf-8-sig")
    assert distribution.set_index("emotion_label")["count"].to_dict() == output.emotion_label.value_counts().to_dict()
    assert abs(distribution.ratio.sum() - 1) <= 1e-5
    summary = json.loads((directory / "blackmyth_clue_positive_summary.json").read_text(encoding="utf-8"))
    assert summary["total_high_confidence_comments"] == len(output)
    assert summary["positive_3cls_count"] == int(positive.sum())
    assert abs(summary["positive_3cls_ratio"] - positive.mean()) <= 1e-5
    return {"input_rows": len(binary), "high_confidence_rows": len(output), "all_output_checks": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local cleaned CSV; never uploaded by this runner.")
    parser.add_argument("--input-encoding", default="utf-8-sig")
    parser.add_argument("--existing-model", required=True)
    parser.add_argument("--binary-model", default="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment")
    parser.add_argument("--base-model", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--run-dir", required=True, help="Must not exist. Use a new directory per run.")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    source = Path(args.input).resolve()
    existing = Path(args.existing_model).resolve()
    protected = [source, *sorted(existing.glob("*.*")), *ROOT.glob("*.csv")]
    before = {str(path): sha256(path) for path in protected}
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HF_DATASETS_CACHE"] = str(run / "datasets-cache")
    if args.local_files_only:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    offline = ["--local-files-only"] if args.local_files_only else []
    report = {"seed": 42, "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script_sha256": {p.name: sha256(p) for p in ROOT.glob("*.py")},
              "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
              "stages": [], "checks": {}, "status": "running"}

    def save_report():
        (run / "run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    def execute(name, script, arguments):
        command = [sys.executable, "-B", "-X", "utf8", str(ROOT / script), *map(str, arguments)]
        started = time.monotonic()
        print(f"START {name}", flush=True)
        log = run / f"{name}.log"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=run, env=env, stdout=stream, stderr=subprocess.STDOUT)
        stage = {"name": name, "command": command, "exit_code": result.returncode, "seconds": round(time.monotonic() - started, 2)}
        report["stages"].append(stage)
        save_report()
        print(f"END {name}: code={result.returncode} seconds={stage['seconds']}", flush=True)
        if result.returncode:
            message = log.read_text(encoding="utf-8")
            print(message[-6000:], flush=True)
            if "out of memory" in message.lower():
                raise MemoryError(name)
            raise RuntimeError(f"{name} failed; see {log}")

    def inference(name, script, arguments, gpu=True):
        for batch in ([8, 4, 2, 1] if gpu else [1]):
            try:
                execute(f"{name}-batch{batch}", script, [*arguments, "--batch-size", batch, "--device", "cuda" if gpu else "cpu"])
                return
            except MemoryError:
                if batch == 1:
                    raise

    try:
        source_df = read_csv(source, args.input_encoding, BASE_COLUMNS)
        if len(source_df) < 200:
            raise ValueError("The smoke run needs at least 200 input comments.")
        sample = source_df[BASE_COLUMNS].sample(n=200, random_state=42).reset_index(drop=True)
        sample_path = run / "sample.csv"
        sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
        report["checks"]["sample_sha256"] = sha256(sample_path)
        for batch in [8, 4, 2, 1]:
            model = run / f"trained-model-batch{batch}"
            try:
                execute(f"train-batch{batch}", TRAINING, ["--model", args.base_model, "--output-dir", model,
                        "--train-per-class", 16, "--eval-per-class", 4, "--max-steps", 5,
                        "--batch-size", batch, "--eval-batch-size", batch, "--device", "cuda", "--seed", 42, *offline])
                break
            except MemoryError:
                if batch == 1:
                    raise
        summary = json.loads((model / "training_summary.json").read_text(encoding="utf-8"))
        assert (summary["train_size"], summary["valid_size"], summary["test_size"], summary["global_step"]) == (112, 28, 28, 5)
        states = list(model.glob("checkpoint-*/trainer_state.json"))
        assert states
        state = json.loads(states[-1].read_text(encoding="utf-8"))
        losses = [item["loss"] for item in state["log_history"] if "loss" in item]
        assert len(losses) == 5 and np.isfinite(losses).all()
        for split in ("valid_metrics", "test_metrics"):
            assert np.isfinite(list(summary[split].values())).all()
        for filename in ("dev_predictions.csv", "test_predictions.csv"):
            predictions = pd.read_csv(model / filename, encoding="utf-8-sig")
            assert len(predictions) == 28
            validate_probabilities(predictions[[f"prob_{label}" for label in LABELS]].values)
        report["checks"]["training"] = {"train_rows": 112, "valid_rows": 28, "test_rows": 28, "steps": 5, "losses": losses, "fp16": summary["fp16"]}
        binary_csv = run / "binary.csv"
        inference("binary-gpu", BINARY, ["--input", sample_path, "--output", binary_csv, "--model", args.binary_model, *offline])
        binary = pd.read_csv(binary_csv, encoding="utf-8-sig")
        assert len(binary) == 200
        pd.testing.assert_frame_equal(binary[BASE_COLUMNS], pd.read_csv(sample_path, encoding="utf-8-sig"))
        validate_probabilities(binary[["negative_prob", "positive_prob"]].values)
        assert binary.pred_label.tolist() == np.where(binary.positive_prob >= binary.negative_prob, "positive", "negative").tolist()
        assert np.allclose(binary.max_prob, binary[["negative_prob", "positive_prob"]].max(axis=1), atol=1e-6, rtol=0)
        # The production flag uses unrounded probabilities; exclude the rounding boundary.
        away = (binary.max_prob - 0.80).abs() > 1e-6
        assert np.array_equal(binary.loc[away, "is_high_confidence"], (binary.loc[away, "max_prob"] >= 0.80).astype(int))
        for name, checkpoint in (("emotion-smoke", model), ("emotion-existing", existing)):
            destination = run / name
            inference(name, EMOTION, ["--input", binary_csv, "--model", checkpoint, "--output-dir", destination])
            report["checks"][name] = check_emotion(binary_csv, destination)

        # Re-infer two confidently classified input rows on CPU with the same real models.
        cpu_input = run / "cpu-input.csv"
        cpu_sample = binary.loc[(binary.max_prob >= 0.90) & (clean_text(binary.comment_text) != ""), BASE_COLUMNS].head(2)
        assert len(cpu_sample) == 2, "Need two confident comments for a CPU chain; threshold is not lowered."
        cpu_sample.to_csv(cpu_input, index=False, encoding="utf-8-sig")
        cpu_binary = run / "cpu-binary.csv"
        inference("binary-cpu", BINARY, ["--input", cpu_input, "--output", cpu_binary, "--model", args.binary_model, *offline], gpu=False)
        cpu_output = pd.read_csv(cpu_binary, encoding="utf-8-sig")
        assert len(cpu_output) == 2
        validate_probabilities(cpu_output[["negative_prob", "positive_prob"]].values)
        inference("emotion-cpu", EMOTION, ["--input", cpu_binary, "--model", existing, "--output-dir", run / "emotion-cpu"], gpu=False)
        report["checks"]["cpu"] = check_emotion(cpu_binary, run / "emotion-cpu")

        # A real tokenizer/model pass (not just a mock) on text exceeding 512 tokens.
        long_input = run / "long-input.csv"
        pd.DataFrame({"comment_text": ["游戏画面很好，但是故事还可以改进。" * 200], "is_high_confidence": [1]}).to_csv(long_input, index=False, encoding="utf-8-sig")
        inference("long-text", EMOTION, ["--input", long_input, "--model", existing, "--output-dir", run / "long-text"])
        report["checks"]["long_text"] = check_emotion(long_input, run / "long-text")
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["checks"]["protected_assets_unchanged"] = all(sha256(path) == digest for path, digest in before.items())
        if not report["checks"]["protected_assets_unchanged"]:
            report["status"] = "failed"
        save_report()
    if report["status"] != "passed":
        raise RuntimeError("Protected asset check failed.")
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
