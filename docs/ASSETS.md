# External assets and rights

The repository contains project code, configuration, synthetic examples and aggregate evidence. It does not distribute base models, trained adapters, raw/processed datasets or tokenizer bundles.

| Asset | Requirement | Provenance / license status |
| --- | --- | --- |
| `train_small.jsonl` | Authorized labeled pairs; original file has 500,000 rows | Source and dataset license unknown; no download link supplied. |
| `test.jsonl` | Unlabeled pairs for inference | 5,000 local rows; source and license unknown. |
| `test_with_model_labels.jsonl` | Original pseudo-label experiment | 5,000 rows; label-producing model and procedure missing. |
| Processed HF datasets | Run preprocessing or privately retain original splits | Derived from above; not distributed. |
| [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | Gated provider access or authorized local model | Llama 3.2 Community License; consult linked provider terms. No Llama weights are included. |
| [SmolLM-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B) | Needed only for archival SmolLM variant | Provider card identifies Apache-2.0; verify conditions when obtaining assets. |
| Final Llama adapters | Train yourself or obtain authorized private copies | Original adapters depend on Llama terms and unknown training-data rights; excluded. |

The archive had no project LICENSE, copyright ownership record, third-party code attribution or dataset citation. A new license was not granted on the user's behalf. Repository privacy does not establish asset rights. Before redistributing data or checkpoints, establish their provenance and applicable terms.

Keep private data under `data/`, adapters under `checkpoints/` or `outputs/`, credentials in the process environment, and downloads under `.local/` if project-local caching is desired. All these locations are ignored. No Git LFS configuration is needed because no large assets are committed.
