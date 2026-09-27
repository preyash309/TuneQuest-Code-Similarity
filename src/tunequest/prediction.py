"""Reconstructed adapter inference. Historical TTA generation was not supplied."""
import csv
import json
import math
from pathlib import Path
from .common import make_prompt, normalize_code


def submission(input_path, output_path, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1].")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    rows, seen = [], set()
    with open(input_path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            identifier, probability = row["id"], float(row["p_final"])
            if identifier in seen or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Duplicate IDs or invalid probabilities.")
            seen.add(identifier)
            rows.append({"id": identifier, "label": int(probability >= threshold)})
    if not rows:
        raise ValueError("Probability CSV is empty.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return {"rows": len(rows), "threshold": threshold, "output": str(output_path)}


def load_classifier(adapter_path, device="cuda"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftConfig, PeftModel
    if device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("CUDA inference requires BF16 support; use --device cpu for a slow FP32 run.")
    config = PeftConfig.from_pretrained(str(adapter_path))
    if config.task_type != "SEQ_CLS":
        raise ValueError("Expected a sequence classification adapter.")
    if not config.base_model_name_or_path:
        raise ValueError("Adapter metadata does not identify a base model.")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name_or_path, num_labels=2,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, str(adapter_path)).to(device).eval()
    return tokenizer, model


def predict(input_path, adapter_path, output_path, max_length=1024, batch_size=8, tta=False, device="cuda"):
    import torch
    if max_length < 1 or batch_size < 1:
        raise ValueError("Max length and batch size must be positive.")
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    records, seen = [], set()
    with open(input_path, encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["id"] is None or row["id"] in seen:
                raise ValueError("IDs must be non-null and unique.")
            seen.add(row["id"])
            for field in ("func1", "func2"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError("Functions must be nonempty strings.")
                row[field] = normalize_code(row[field])
            records.append(row)
    if not records:
        raise ValueError("Input is empty.")
    tokenizer, model = load_classifier(adapter_path, device)
    results = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        def score(reverse=False):
            prompts = [make_prompt(r["func2"], r["func1"]) if reverse else make_prompt(r["func1"], r["func2"]) for r in batch]
            inputs = tokenizer(prompts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                return torch.softmax(model(**inputs).logits.float(), dim=-1)[:, 1].cpu().tolist()
        forward = score()
        reverse = score(True) if tta else forward
        results.extend({"id": r["id"], "p_forward": a, "p_reverse": b, "p_final": (a + b) / 2} for r, a, b in zip(batch, forward, reverse))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["id", "p_forward", "p_reverse", "p_final"])
        writer.writeheader()
        writer.writerows(results)
    return {"rows": len(results), "tta": tta, "output": str(output_path)}
