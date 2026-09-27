# Reproducibility and verification

Verification date: 2026-09-27. Host: Windows, Python 3.11.0, CUDA-capable GPU with BF16 support. The supplied project environment successfully imported the core ML dependencies and passed `python -m pip check`.

| Direct dependency | Observed version |
| --- | --- |
| torch | 2.11.0+cu128 |
| transformers | 5.15.0 |
| peft | 0.20.0 |
| datasets | 5.0.1 |
| accelerate | 1.14.0 |
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| safetensors | 0.8.0 |
| bitsandbytes (archive only) | 0.50.1 |

These versions come from the supplied environment. A fresh network installation of every pinned ML package was not independently reproduced. The editable project package was installed successfully without downloading dependencies. The pinned ML extra captures direct dependencies, not every transitive dependency or CUDA driver. There was no original lockfile or pinned model revision. Historical runs cannot be claimed bit-for-bit reproducible.

## Executed checks

| Check | Outcome and scope |
| --- | --- |
| Unit/data/adapter suite | 12 tests passed, no skips in the supplied ML environment. Tests cover prompts, normalization, metrics, sweep tie behavior, config CLI, submission threshold equality, invalid input, duplicate/conflicting pairs, deterministic sampling, symmetry, disk dataset roundtrip, local adapter loading and TTA. |
| Golden 5k preprocessing | Executed archived `pre.py` with only input/output locations changed; compared all five columns and row order against supported preprocessing. All 4,500 training and 250 validation rows identical. |
| Historical submissions | Generated submissions from supplied `test_probabilities_tta.csv` at .35, .37 and .38. All IDs and labels matched each original 5,000-row CSV. |
| Saved validation metrics | Loaded BF16 NPY arrays with pickle disabled. Recomputed .5/.06 metrics and the .05..95 threshold sweep. Selected .06, F1 0.5416667. Labels are model-generated. |
| CUDA trainer smoke | Random one-layer tiny Llama classifier, vocabulary6/hidden16/intermediate32, rank2/alpha4; synthetic16 training and4 validation pairs, max-length64/warmup0. Preserved trainer completed one epoch, saved adapter/tokenizer/arrays, reloaded adapter, predicted both orientations, evaluated saved arrays. Passed. This is plumbing validation only. |
| Imports/dependencies | torch, NumPy, pandas, datasets, transformers, PEFT and sklearn imported; `pip check` found no conflicts. |
| CLI/config | Help, config check, preprocessing disk roundtrip, training smoke, prediction, evaluation and model-free submission exercised. |
| Syntax | `python -m compileall -q src tests` passed. |

The original `test.py` is a token audit and `testingtta.py` is submission generation. Neither is an automated test suite; no preexisting automated tests were found.

Run the suite after installation:

```text
python -m unittest discover -s tests -v
python -m tunequest train --check-config
python -m pip check
```

Without optional ML/data dependencies the dependent tests skip explicitly. The GitHub workflow tests the lightweight core and data preprocessing jobs. It does not train or download a base model, and does not run the optional adapter test without ML dependencies.

## Controls and unresolved limits

Normalization and prompt text match the originals. Cleaning precedes the split; conflicts remove all instances of the unordered pair; deduplication keeps the first original orientation. The split uses scikit-learn stratification with random state42. Sampling occurs after the fixed validation split, and reversal is applied only to training. Labels are not added to prompts. The data audit checks IDs and unordered pairs, not shared functions or source repositories.

The original model's classification head is randomly initialized then trained alongside LoRA and saved as a PEFT module. Loading an adapter uses its base-model identifier and restores that head. The tiny CPU and CUDA tests exercised this route. No full Llama base model was present inside the supplied project, so full-model compatibility, historical numerical predictions and full training quality remain unverified. Original binary `training_args.bin` files were preserved but not unpickled.

Always record dataset source/license, label provenance, model revision, dependency versions, split IDs, seed, hardware, configuration and output metrics for new experiments. Keep original validation sets intact. New ground-truth training runs do not reproduce the historical pseudo-label checkpoint merely by sharing its model settings.

Local verification logs and golden outputs are retained in ignored `.local/reconstruction/`; the full original backup is `.local/original/`. These private records and all raw data remain outside Git.
