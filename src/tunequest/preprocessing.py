"""Stratified pair-level preprocessing compatible with the sampled experiments."""
import json
from pathlib import Path
from .common import make_prompt, normalize_code


def prepare_frames(df, sample_size=None, validation_fraction=0.05, seed=42, symmetry=False):
    from sklearn.model_selection import train_test_split
    import pandas as pd

    required = {"id", "func1", "func2", "label"}
    if missing := required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("IDs must be non-null and unique before cleaning.")
    raw_rows = len(df)
    df = df.dropna(subset=["func1", "func2", "label"]).copy()
    # Fail instead of silently truncating fractional labels with astype(int).
    if not df["label"].isin([0, 1]).all():
        raise ValueError("Labels must be numeric 0 or 1.")
    df["label"] = df["label"].astype(int)
    for col in ("func1", "func2"):
        df[col] = df[col].map(normalize_code)
    df = df[(df.func1.str.len() > 0) & (df.func2.str.len() > 0)].copy()
    df["pair_key"] = [tuple(sorted((a, b))) for a, b in zip(df.func1, df.func2)]
    counts = df.groupby("pair_key")["label"].nunique()
    conflicts = set(counts[counts > 1].index)
    df = df[~df.pair_key.isin(conflicts)].drop_duplicates("pair_key", keep="first")
    clean_rows = len(df)
    train, val = train_test_split(df, test_size=validation_fraction, stratify=df.label, random_state=seed)
    train, val = train.reset_index(drop=True), val.reset_index(drop=True)
    pool_size = len(train)
    if sample_size is not None:
        if sample_size < 1:
            raise ValueError("Sample size must be positive.")
        if sample_size < len(train):
            train, _ = train_test_split(train, train_size=sample_size, stratify=train.label, random_state=seed)
            train = train.reset_index(drop=True)
    if set(train.pair_key) & set(val.pair_key) or set(train.id) & set(val.id):
        raise ValueError("Train/validation leakage detected.")
    train_before = len(train)
    if symmetry:
        reverse = train.copy()
        reverse["func1"], reverse["func2"] = train.func2, train.func1
        train["orientation"] = "original"
        reverse["orientation"] = "reversed"
        train = pd.concat([train, reverse], ignore_index=True)
    for frame in (train, val):
        frame["prompt"] = [make_prompt(a, b) for a, b in zip(frame.func1, frame.func2)]
        frame.drop(columns="pair_key", inplace=True)
    train = train.sample(frac=1, random_state=seed).reset_index(drop=True)
    cols = ["id", "func1", "func2", "prompt", "label"]
    if symmetry:
        train = train[cols + ["orientation"]]
    else:
        train = train[cols]
    val = val[cols]
    report = {
        "raw_rows": raw_rows, "clean_unique_pairs": clean_rows,
        "conflicting_unordered_pairs": len(conflicts), "full_training_pool": pool_size,
        "training_rows_before_symmetry": train_before, "training_rows": len(train),
        "validation_rows": len(val), "seed": seed, "validation_fraction": validation_fraction,
        "symmetry_augmentation": symmetry, "train_validation_id_overlap": 0,
        "train_validation_pair_overlap": 0,
        "label_distribution_train": {str(k): int(v) for k, v in train.label.value_counts().items()},
        "label_distribution_validation": {str(k): int(v) for k, v in val.label.value_counts().items()},
    }
    return train, val, report


def preprocess(input_path, output_path, **kwargs):
    import pandas as pd
    from datasets import Dataset, DatasetDict
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output_path}")
    train, val, report = prepare_frames(pd.read_json(input_path, lines=True), **kwargs)
    dataset = DatasetDict({"train": Dataset.from_pandas(train, preserve_index=False),
                           "validation": Dataset.from_pandas(val, preserve_index=False)})
    dataset.save_to_disk(str(output_path))
    report["input_file"] = str(input_path)
    (output_path / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
