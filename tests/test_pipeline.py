"""Regression tests with synthetic data only; no model download required."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline_runtime as runtime


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


binary = load("binary_test", "情感二分类正负推理.py")
emotion = load("emotion_test", "用于情感多分类推理任务的训练后的RoBERTa模型.py")
training = load("training_test", "RoBERTa模型训练_FP16.py")


class PipelineTests(unittest.TestCase):
    def test_default_training_paths_are_repo_relative(self):
        for split in ("train", "valid", "test"):
            path = Path(getattr(training.Config(), split + "_file"))
            self.assertTrue(path.is_absolute())
            self.assertTrue(path.is_file())

    def test_binary_to_emotion_encoding_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "中文.csv"
            pd.DataFrame({"comment_text": ["中文评论🙂"], "is_high_confidence": [1]}).to_csv(path, index=False, encoding="utf-8-sig")
            args = emotion.parse_args([])
            df = runtime.read_csv(path, args.encoding, ["comment_text", "is_high_confidence"])
            self.assertEqual(df.comment_text.iloc[0], "中文评论🙂")

    def test_explicit_legacy_encoding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.csv"
            pd.DataFrame({"comment_text": ["旧版中文"]}).to_csv(path, index=False, encoding="gb18030")
            self.assertEqual(runtime.read_csv(path, "gb18030", ["comment_text"]).iloc[0, 0], "旧版中文")

    def test_missing_file_and_columns_fail_before_model_load(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(binary.AutoTokenizer, "from_pretrained") as loader:
            with self.assertRaises(FileNotFoundError):
                binary.main(["--input", str(Path(folder) / "missing.csv")])
            path = Path(folder) / "missing_column.csv"
            pd.DataFrame({"comment_text": ["测试"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                binary.main(["--input", str(path)])
            loader.assert_not_called()

    def test_empty_binary_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "empty.csv"
            pd.DataFrame(columns=["comment_date", "comment_text", "like_count", "reply_count"]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "no comments"):
                binary.main(["--input", str(path)])

    def test_null_and_blank_comments(self):
        df = pd.DataFrame({"comment_text": [None, np.nan, "  ", " 正常 "], "is_high_confidence": [1, 1, 1, 1]})
        actual = runtime.filter_high_confidence(df, "comment_text", "is_high_confidence")
        self.assertEqual(actual.comment_text.tolist(), ["正常"])

    def test_no_high_confidence_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(emotion.AutoTokenizer, "from_pretrained") as loader:
            path = Path(folder) / "low.csv"
            pd.DataFrame({"comment_text": ["测试"], "is_high_confidence": [0]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "No high-confidence"):
                emotion.main(["--input", str(path)])
            loader.assert_not_called()

    def test_invalid_confidence_flag(self):
        for value in (2, "bad", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "only 0 or 1"):
                runtime.filter_high_confidence(pd.DataFrame({"text": ["测试"], "flag": [value]}), "text", "flag")

    def test_binary_mapping_and_reversed_chinese_order(self):
        self.assertEqual(binary.build_label_mapping({0: "Negative", 1: "Positive"}), {0: "negative", 1: "positive"})
        self.assertEqual(binary.build_label_mapping({0: "满意", 1: "不满意"}), {0: "positive", 1: "negative"})
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            binary.build_label_mapping({0: "LABEL_0", 1: "LABEL_1"})

    def test_emotion_mapping_consistency_and_config_fallback(self):
        label2id = {label: i for i, label in enumerate(runtime.LABELS)}
        id2label = {i: label for label, i in label2id.items()}
        config = SimpleNamespace(label2id=label2id, id2label=id2label, num_labels=7)
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(emotion.load_label_mapping(folder, config), (label2id, id2label))
            altered = dict(id2label)
            altered[0], altered[1] = altered[1], altered[0]
            Path(folder, "label_mapping.json").write_text(json.dumps({"label2id": {v: k for k, v in altered.items()}, "id2label": altered}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                emotion.load_label_mapping(folder, config)

    def test_prediction_report_with_absent_classes(self):
        fake_trainer = SimpleNamespace(predict=lambda _: SimpleNamespace(predictions=np.array([[9., 0, 0, 0, 0, 0, 0]])))
        with tempfile.TemporaryDirectory() as folder:
            csv_path, report_path = Path(folder, "pred.csv"), Path(folder, "report.json")
            training.export_predictions(fake_trainer, None, pd.DataFrame({"labels": [0]}), csv_path, report_path, training.CFG.id2label)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["sadness"]["support"], 1)
            self.assertEqual(report["fear"]["support"], 0)

    def test_stratified_sample_is_deterministic(self):
        df = pd.DataFrame([{"label": label, "content": str(i)} for label in runtime.LABELS for i in range(20)])
        a = runtime.sample_per_class(df, 16, 42)
        pd.testing.assert_frame_equal(a, runtime.sample_per_class(df, 16, 42))
        self.assertEqual(a.label.value_counts().tolist(), [16] * 7)
        with self.assertRaisesRegex(ValueError, "Need"):
            runtime.sample_per_class(df, 21, 42)

    def test_missing_training_class(self):
        with self.assertRaisesRegex(ValueError, "missing classes"):
            training.compute_class_weights([0, 1], 7)

    def test_cpu_disables_fp16(self):
        cfg, args = training.configure(["--device", "cpu"])
        self.assertFalse(cfg.fp16)
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            training.configure(["--device", "cpu", "--precision", "fp16"])

    def test_probability_validation(self):
        runtime.validate_probabilities([[0.4, 0.6]])
        for probs in ([[float("nan"), 1]], [[-0.1, 1.1]], [[0.1, 0.1]]):
            with self.assertRaises(ValueError):
                runtime.validate_probabilities(probs)

    def test_outputs_cannot_overwrite_inputs(self):
        with self.assertRaisesRegex(ValueError, "overwrite"):
            runtime.prepare_output("same.csv", ["same.csv"])

    def test_existing_training_model_is_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "config.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                training.main(["--output-dir", folder])

    def test_long_text_passes_truncation_limit(self):
        seen = {}
        def tokenizer(texts, **kwargs):
            seen.update(kwargs)
            return {"input_ids": torch.ones((1, 128), dtype=torch.long)}
        model = lambda **_: SimpleNamespace(logits=torch.zeros((1, 7)))
        ids, scores, probs = emotion.predict_batch(["中文" * 2000], tokenizer, model, "cpu", 128)
        self.assertTrue(seen["truncation"])
        self.assertEqual(seen["max_length"], 128)
        self.assertEqual(probs.shape, (1, 7))


if __name__ == "__main__":
    unittest.main()
