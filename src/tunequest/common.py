"""Model-independent prompt and probability utilities."""
import math


def normalize_code(code):
    """Match historical whitespace normalization; preserve internal indentation."""
    if not isinstance(code, str):
        code = str(code)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in code.split("\n")).strip()


def make_prompt(func1, func2):
    return (
        "Determine if Function 1 and Function 2 are semantically equivalent.\n"
        f"### Function 1:\n{func1}\n"
        f"### Function 2:\n{func2}\n"
        "### Equivalent:"
    )


def binary_metrics(labels, probabilities, threshold):
    labels, probabilities = list(labels), list(probabilities)
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must be nonempty and equal length.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1].")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("Labels must be binary.")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Probabilities must be finite and in [0, 1].")
    pred = [p >= threshold for p in probabilities]
    tp = sum(y == 1 and p for y, p in zip(labels, pred))
    fp = sum(y == 0 and p for y, p in zip(labels, pred))
    fn = sum(y == 1 and not p for y, p in zip(labels, pred))
    return {
        "samples": len(labels), "threshold": threshold,
        "accuracy": sum(y == p for y, p in zip(labels, pred)) / len(labels),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def best_threshold(labels, probabilities):
    """Historical 0.05..0.95 sweep, first strict F1 improvement wins."""
    labels, probabilities = list(labels), list(probabilities)
    best = binary_metrics(labels, probabilities, 0.5)
    best_f1 = 0.0
    for step in range(181):
        candidate = binary_metrics(labels, probabilities, round(0.05 + step * 0.005, 6))
        if candidate["f1"] > best_f1:
            best, best_f1 = candidate, candidate["f1"]
    return best
