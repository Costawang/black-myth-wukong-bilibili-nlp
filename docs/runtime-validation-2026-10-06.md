# Runtime validation — 2026-10-06

This is a software runtime check of the three NLP scripts, not a new research result or an assessment of sentiment accuracy. The baseline is commit `4030e9ebda29b0fc83913bd0bbc00f6b9ae96f16` on `main`. Code changes were tested in a separate checkout; the original source archive, historical results and trained model were retained.

## Reproduced problems and fixes

| Before | After |
| --- | --- |
| Training looks for `train.txt` in the working directory and raises `FileNotFoundError` from the repository root. | Defaults locate the bundled CLUE directory relative to the repository; CLI overrides are available. |
| Binary output uses UTF-8-SIG; emotion inference reads GB18030, corrupting Chinese text. | New stage inputs/outputs use UTF-8-SIG consistently; legacy encoding requires an explicit option. |
| Binary labels `{0: "满意", 1: "不满意"}` are incorrectly mapped to negative/positive by substring matching and fallback. | Exact aliases honor the declared mapping; ambiguous semantics are rejected. |
| Exporting a seven-class report with a one-class evaluation sample raises `ValueError`. | All seven IDs are explicitly supplied to the report, including zero-support classes. |

Further guards check files/columns before model loading, normalize null text, validate confidence flags and saved model labels, reject non-finite probabilities/losses, protect existing training output directories, and support CPU execution without FP16. Research label definitions and the 0.80 confidence threshold are unchanged.

## Verified environment

- Windows, Python 3.11.9; NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12 GB VRAM, driver 616.92.
- PyTorch 2.11.0+cu128; Transformers 4.37.2; Datasets 2.18.0; Accelerate 0.27.2.
- NumPy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, PyArrow 23.0.1.
- `pip check`: no broken requirements. Dependency versions are captured in the Windows/CUDA lock file.
- Base model: `hfl/chinese-roberta-wwm-ext`, cached snapshot `5c58d0b8ec1d9014354d691c538661bf00bfdb44`.
- Binary model: `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`, cached snapshot `7c257c5cde3225d0789acfa8d67eb043289b0295`.
- Existing seven-class checkpoint loaded from the owner's local model directory; weights are not uploaded.

The installed environment was reused without changing packages. These results do not certify a fresh installation or other operating systems. Legacy library deprecation notices occurred; they did not prevent training/inference. The early-stopping callback also logs a missing `eval_macro_f1` notice during the separate `test_*` evaluation; test metrics and reports were produced successfully after training had finished.

## Results

| Check | Result | Process elapsed time |
| --- | --- | ---: |
| Synthetic regression suite | 18 tests passed | 0.082 s test body, excluding imports |
| GPU training | 112 train / 28 validation / 28 test rows; 5 steps, FP16, batch 8 | 27.56 s |
| GPU binary inference | 200 input rows → 200 output rows | 18.70 s |
| Newly saved model: seven-class inference | 137 high-confidence non-empty comments | 20.42 s |
| Existing model: seven-class inference | The same 137 comments | 20.48 s |
| CPU binary inference | 2 comments, batch 1 | 5.77 s |
| CPU existing-model emotion inference | 2 comments, batch 1 | 5.67 s |
| Real-model long-text test | 1 synthetic comment exceeding 512 tokens, successfully truncated | 19.80 s |

GPU inference used batch 8 without an OOM retry. Seed 42 was used for comment sampling and class-balanced train/evaluation samples. Five finite training losses were recorded: `2.0966, 1.9493, 2.1373, 1.9286, 2.0022`. The model, tokenizer, seven-label mapping and evaluation exports were saved successfully; loading the saved checkpoint for inference verified the training-to-inference boundary.

All output checks passed: preserved input text/order, expected retained row counts, labels consistent with probabilities, probabilities finite and within [0, 1], row sums within 1e-5 of 1, per-class JSON scores consistent with CSV columns, and distribution/positive-emotion summaries consistent with detail rows. CPU and long-text outputs passed the same emotion checks. Asset hashes checked by the smoke runner remained unchanged.

The 18 regression tests cover path defaults, encoding round trips and legacy encoding, missing files/columns, empty CSV, null/blank comments, no high-confidence rows, invalid confidence flags, reversed/ambiguous binary labels, inconsistent emotion mappings and config fallback, missing evaluation/training classes, deterministic sampling, CPU precision, invalid probabilities, input/output collisions, protected training directories and long-text truncation parameters.

## Reproduction and limits

Use the commands in [README](../README.md#automated-regression-and-real-model-smoke-checks). The full local report preserves command arguments, environment, code fingerprints, timing and failure status. The public report intentionally excludes raw comments, private paths, model weights and generated CSV files.

Neither full five-epoch training nor all-comment inference was performed. Small-sample success does not establish full-run memory usage, exact reproduction of historical outputs, or model quality. The short-trained checkpoint is only a runtime fixture. No performance claims are made from its metrics. Existing production-model inference was checked separately; its historical training metrics were not re-evaluated in this task.
