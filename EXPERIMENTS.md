# Research history and evidence

The archive preserves the supplied scripts byte-for-byte. They execute at import time, use working-directory paths, can overwrite outputs, and are unsupported archival code. Inspect them before use. The supported CLI replaces the primary workflow; it does not retroactively make the experiments a controlled ablation study.

| Experiment | Original implementation | Method and evidence | Status |
| --- | --- | --- | --- |
| Symmetric dataset | [preprocess_data.py](experiments/archive/preprocess_data.py) | Normalize; SHA256 unordered-pair dedup; remove conflicts; stratified 95/5 split seed 42; reverse training pairs only. Report: 943,670 augmented train, 24,834 validation. | Historical report retained. New CLI offers reversal; full 500k rerun not performed. |
| SmolLM QLoRA | [training.py](experiments/archive/training.py), [test.py](experiments/archive/test.py) | SmolLM-1.7B, NF4 double quantization, rank 32/alpha64, all linear targets, score head, 1,536 tokens; batch4/accum4; 100 max steps; warmup1770. Token audit samples up to 10k prompts. | No saved final adapter or metric evidence found; empty output directory is not evidence of success. Token audit is not a test suite. |
| Sampled data | [preprocess_new.py](experiments/archive/preprocess_new.py) | Unordered tuple dedup, stratified split then stratified sample, no reversal. `processed_codeclone_250k` report: 250,000 train, 24,834 validation. | Script currently names pseudo-label input and `processed_codeclone_50k` output; it cannot reproduce the 250k artifact using its saved defaults. Supported CLI reconstructs intended configurable method. |
| Llama QLoRA | [training_new.py](experiments/archive/training_new.py) | Llama-3.2-1B, NF4 without double quantization, attention projection LoRA r16/alpha64, score head, 768 tokens, batch8/accum4; 100 max steps, warmup470. | Adapter metadata and local weights exist. Final evaluation arrays/logs absent. Although imported, `prepare_model_for_kbit_training` is not called in this variant. Not independently rerun. |
| Pseudo-label preprocessing | [pre.py](experiments/archive/pre.py) | `test_with_model_labels.jsonl`, 5,000 rows; 4,750 train pool, sample4,500; validation250; seed42; no reversal. | All output columns and row order independently matched by supported preprocessing on the supplied input. |
| Llama BF16 LoRA | [training_new2.py](experiments/archive/training_new2.py) | Llama-3.2-1B BF16, same attention LoRA r16/alpha64, score head, 1,024 tokens, batch2/accum8, 1 epoch, warmup470. | Final adapter, tokenizer, validation arrays and threshold0.06 exist. Callable supported trainer preserves expressions and settings. Saved-array metrics recomputed; training only smoke-tested on a random tiny model. |
| Submission thresholds | [testingtta.py](experiments/archive/testingtta.py) | Converts `p_final` CSV to binary CSV; saved submissions at .35, .37 and .38. | No leaderboard scores or inference producer supplied. Supported submission command reproduces all three saved files. |

## Available evidence

[experiments/evidence](experiments/evidence) contains the three original preprocessing reports, both Llama adapter configurations, the saved BF16 threshold and recomputed metrics. Large evidence remains in the original local tree and its backup, excluded from publication. The metadata identifies the base model but no pinned revision.

The BF16 experiment appears to study learning from model labels. Its filename, preprocessing report and dataset labels establish that the labels are model-generated; there is no teacher script or teacher identity. Do not assert knowledge distillation quality or human-label generalization. No evidence supports a conclusion that BF16 outperforms QLoRA, that reversal improves accuracy, or that any threshold improves leaderboard scores.

## Reproduction

Use `python -m tunequest preprocess --input data/train.jsonl --output data/processed --sample-size 250000` for the sampled ground-truth path, or add `--symmetry` to augment training only. Each run must have its own output path. Sampled and augmented methods are alternatives; compare on the same fixed validation protocol. Oversized sampling now uses the full pool without calling scikit-learn with an invalid train size.

To study the original pseudo-label BF16 run, use the privately retained `test_with_model_labels.jsonl`, sample4,500 and `configs/bf16.json`. Its origin and redistribution rights must be established first. Do not use this validation split as a genuine held-out benchmark.

The metric evaluator's sweep uses exact decimal grid thresholds .05 through .95 in increments of .005. Original training uses NumPy's floating-point `arange`; only inputs exactly at floating-point boundary differences may differ. The historical 250-example best threshold and metrics match.
