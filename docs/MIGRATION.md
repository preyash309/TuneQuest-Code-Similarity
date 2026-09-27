# Migration map

All original files were preserved locally before edits in an independent 44,481-file backup, including the original environment and generated artifacts. No original research evidence was deleted. The inventory and private reconnaissance report live in ignored `.local/reconstruction/`. Source copies under `experiments/archive/` retain original bytes. The supported package uses `src/` to avoid a Windows case-insensitive collision between `TuneQuest` and `tunequest`.

| Original location under `TuneQuest/` | Repository location | Reason |
| --- | --- | --- |
| `preprocess_data.py` | `experiments/archive/preprocess_data.py`; supported `src/tunequest/preprocessing.py --symmetry` | Preserve training reversal alternative; expose reusable preprocessing. |
| `preprocess_new.py`, `pre.py` | Same filenames in archive; supported `src/tunequest/preprocessing.py` | Consolidate identical cleaning, prompt, split and sampling logic. |
| `training_new2.py` | Archive; supported `src/tunequest/training.py` and `configs/bf16.json` | Defer execution, configure portable paths and preserve training computations. |
| `training.py`, `training_new.py` | Same filenames in archive | Retain materially different SmolLM/Llama QLoRA variants. |
| `test.py` | `experiments/archive/test.py` | Identify correctly as exploratory token-length audit. |
| `testingtta.py` | Archive; supported `src/tunequest/prediction.py` submission function | Replace repeated hardcoded threshold edits with an argument. |
| `tempCodeRunnerFile.py` | Local original and backup only | Single undefined scratch identifier, no scientific content. |
| Three `processed*/preprocessing_report.json` | `experiments/evidence/processed*.json` | Preserve small aggregate historical evidence without datasets. |
| Two final `adapter_config.json` files | `experiments/evidence/*_adapter.json` | Preserve model architecture/configuration evidence without weights. |
| BF16 `best_threshold.txt` | `experiments/evidence/bf16_best_threshold.txt` | Preserve threshold selection evidence. |
| JSONL, Arrow, validation NPY, tokenizers, adapters, CSV outputs, environment | Original local locations and `.local/original/`, ignored | Unknown provenance, large data, derived outputs and machine-specific binaries. |

## Recorded behavior changes

- Supported code is import-safe; archival scripts remain unmodified.
- CLI settings and paths replace hardcoded script constants; default dataset path is `data/processed`, output `outputs/llama_bf16_lora`.
- Invalid fractional/nonbinary labels and null IDs now fail rather than coercing labels or silently filtering them. Missing function/label rows and empty functions are still removed before split.
- Oversized or full-pool sampling uses the entire pool; the originals can fail when `train_size == len(train)`.
- Output guards prevent accidental replacement of existing datasets/training runs/submissions.
- BF16 training explicitly checks BF16 hardware; the original always loads BF16 weights even when its Trainer falls back to FP16.
- Prompt text, whitespace normalization, conflict removal, unordered deduplication, seed42 stratified split, sampled row ordering and BF16 computation order are preserved for valid sampled inputs. Golden comparison confirms exact compatibility on the original 5k input.
- Prediction and probability averaging are reconstructed because their historical producer is missing. Historical TTA aggregation cannot be inferred solely from the output file.
- Archived QLoRA choices and warmup counts were not silently corrected. They are documented limitations.
