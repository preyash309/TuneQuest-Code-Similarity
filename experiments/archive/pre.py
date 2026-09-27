import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = "test_with_model_labels.jsonl"

OUTPUT = "processed_codeclone_5k"

TRAIN_SAMPLE_SIZE = 4_500

VAL_FRACTION = 0.05

RANDOM_STATE = 42


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

print(f"Input file: {INPUT}")

df = pd.read_json(
    INPUT,
    lines=True
)

print(f"Raw rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 2. COLUMN VALIDATION
# ============================================================

required_columns = {
    "id",
    "func1",
    "func2",
    "label"
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("\nRequired columns found:")
for col in required_columns:
    print(f"  {col}")


# ============================================================
# 3. ID AUDIT
# ============================================================

print("\n" + "=" * 70)
print("2. ID AUDIT")
print("=" * 70)

unique_ids = df["id"].nunique()

duplicate_ids = len(df) - unique_ids

print(f"Unique IDs:    {unique_ids:,}")
print(f"Duplicate IDs: {duplicate_ids:,}")

if duplicate_ids > 0:

    raise ValueError(
        "Duplicate IDs detected. "
        "Stop and investigate before training."
    )


# ============================================================
# 4. BASIC VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("3. BASIC VALIDATION")
print("=" * 70)

before = len(df)

df = df.dropna(
    subset=[
        "func1",
        "func2",
        "label"
    ]
).copy()

print(
    f"Removed rows with missing values: "
    f"{before - len(df):,}"
)


# Convert labels to integers

df["label"] = df["label"].astype(int)

invalid_labels = ~df["label"].isin([0, 1])

invalid_count = invalid_labels.sum()

if invalid_count > 0:

    print(
        f"Removing invalid labels: "
        f"{invalid_count:,}"
    )

    df = df[
        ~invalid_labels
    ].copy()

else:

    print(
        "Removed rows with invalid labels: 0"
    )


# ============================================================
# 5. CODE NORMALIZATION
# ============================================================

print("\n" + "=" * 70)
print("4. CODE NORMALIZATION")
print("=" * 70)


def normalize_code(code):

    if not isinstance(code, str):
        code = str(code)

    # Normalize line endings
    code = code.replace(
        "\r\n",
        "\n"
    )

    code = code.replace(
        "\r",
        "\n"
    )

    # Remove trailing whitespace
    # while preserving indentation.
    code = "\n".join(
        line.rstrip()
        for line in code.split("\n")
    )

    # Remove leading/trailing blank space
    return code.strip()


print("Normalizing func1...")

df["func1"] = df["func1"].map(
    normalize_code
)

print("Normalizing func2...")

df["func2"] = df["func2"].map(
    normalize_code
)

print("Normalization complete.")


# ============================================================
# 6. EMPTY FUNCTION CHECK
# ============================================================

print("\n" + "=" * 70)
print("5. EMPTY FUNCTION CHECK")
print("=" * 70)

before = len(df)

df = df[
    (df["func1"].str.len() > 0)
    &
    (df["func2"].str.len() > 0)
].copy()

print(
    f"Removed empty-function rows: "
    f"{before - len(df):,}"
)


# ============================================================
# 7. UNORDERED PAIR KEY
# ============================================================

print("\n" + "=" * 70)
print("6. PAIR-LEVEL DEDUPLICATION")
print("=" * 70)

print("Creating unordered pair keys...")


df["pair_key"] = list(
    zip(
        df[["func1", "func2"]].min(axis=1),
        df[["func1", "func2"]].max(axis=1)
    )
)

print("Pair keys created.")


# ============================================================
# 8. CONTRADICTORY LABEL CHECK
# ============================================================

print("\n" + "=" * 70)
print("7. CONTRADICTORY LABEL CHECK")
print("=" * 70)

label_counts = (
    df.groupby("pair_key")["label"]
    .nunique()
)

conflicting_keys = set(
    label_counts[
        label_counts > 1
    ].index
)

print(
    f"Conflicting unordered pairs: "
    f"{len(conflicting_keys):,}"
)


if conflicting_keys:

    before = len(df)

    df = df[
        ~df["pair_key"].isin(
            conflicting_keys
        )
    ].copy()

    print(
        f"Rows removed due to conflicts: "
        f"{before - len(df):,}"
    )

else:

    print(
        "No contradictory unordered pairs."
    )


# ============================================================
# 9. EXACT / REVERSED DEDUPLICATION
# ============================================================

print("\n" + "=" * 70)
print("8. EXACT / REVERSED PAIR DEDUPLICATION")
print("=" * 70)

before = len(df)

df = df.drop_duplicates(
    subset=["pair_key"],
    keep="first"
).copy()

print(
    f"Removed duplicate/reversed pairs: "
    f"{before - len(df):,}"
)

print(
    f"Unique unordered pairs remaining: "
    f"{len(df):,}"
)


# Remove temporary key

df = df.drop(
    columns=["pair_key"]
)


# ============================================================
# 10. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("9. LABEL DISTRIBUTION")
print("=" * 70)

label_counts = df["label"].value_counts()

label_percentages = (
    df["label"]
    .value_counts(
        normalize=True
    )
    * 100
)

for label in [0, 1]:

    print(
        f"Label {label}: "
        f"{label_counts.get(label, 0):,} "
        f"({label_percentages.get(label, 0):.2f}%)"
    )


# ============================================================
# 11. GOLDEN TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 70)
print("10. TRAIN / VALIDATION SPLIT")
print("=" * 70)

print(
    f"Validation fraction: "
    f"{VAL_FRACTION:.2%}"
)

train_df, val_df = train_test_split(
    df,

    test_size=VAL_FRACTION,

    stratify=df["label"],

    random_state=RANDOM_STATE
)

train_df = train_df.reset_index(
    drop=True
)

val_df = val_df.reset_index(
    drop=True
)

print(
    f"Full training pool: "
    f"{len(train_df):,}"
)

print(
    f"Validation: "
    f"{len(val_df):,}"
)


# ============================================================
# 12. VERIFY ID SEPARATION
# ============================================================

print("\n" + "=" * 70)
print("11. TRAIN / VALIDATION ID AUDIT")
print("=" * 70)

train_ids = set(
    train_df["id"]
)

val_ids = set(
    val_df["id"]
)

overlap = train_ids.intersection(
    val_ids
)

print(
    f"Train IDs: "
    f"{len(train_ids):,}"
)

print(
    f"Validation IDs: "
    f"{len(val_ids):,}"
)

print(
    f"ID overlap: "
    f"{len(overlap):,}"
)

if overlap:

    raise ValueError(
        "TRAIN/VALIDATION ID LEAKAGE DETECTED!"
    )

print(
    "No train/validation ID overlap."
)


# ============================================================
# 13. STRATIFIED TRAINING SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("12. CREATING TRAINING SET")
print("=" * 70)

print(
    f"Available unique training pairs: "
    f"{len(train_df):,}"
)

print(
    f"Requested training pairs: "
    f"{TRAIN_SAMPLE_SIZE:,}"
)


if TRAIN_SAMPLE_SIZE > len(train_df):

    print(
        f"Requested {TRAIN_SAMPLE_SIZE:,} training pairs, "
        f"but only {len(train_df):,} are available."
    )

    print(
        f"Using all {len(train_df):,} available training pairs instead."
    )

    TRAIN_SAMPLE_SIZE = len(train_df)


# Stratified sampling

sample_train_df, _ = train_test_split(
    train_df,

    train_size=TRAIN_SAMPLE_SIZE,

    stratify=train_df["label"],

    random_state=RANDOM_STATE
)

sample_train_df = sample_train_df.reset_index(
    drop=True
)


print(
    f"Final training pairs: "
    f"{len(sample_train_df):,}"
)


# ============================================================
# 14. VERIFY SAMPLE LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("13. TRAINING SAMPLE LABEL DISTRIBUTION")
print("=" * 70)

sample_counts = (
    sample_train_df["label"]
    .value_counts()
)

sample_percentages = (
    sample_train_df["label"]
    .value_counts(
        normalize=True
    )
    * 100
)

for label in [0, 1]:

    print(
        f"Label {label}: "
        f"{sample_counts.get(label, 0):,} "
        f"({sample_percentages.get(label, 0):.2f}%)"
    )


# ============================================================
# 15. VERIFY SAMPLE HAS NO VALIDATION OVERLAP
# ============================================================

print("\n" + "=" * 70)
print("14. SAMPLE / VALIDATION LEAKAGE CHECK")
print("=" * 70)

sample_ids = set(
    sample_train_df["id"]
)

sample_val_overlap = sample_ids.intersection(
    val_ids
)

print(
    f"Sample/validation ID overlap: "
    f"{len(sample_val_overlap):,}"
)

if sample_val_overlap:

    raise ValueError(
        "TRAINING SAMPLE OVERLAPS VALIDATION!"
    )

print(
    "No sample/validation ID overlap."
)


# ============================================================
# 16. CREATE COMMON PROMPT
# ============================================================

print("\n" + "=" * 70)
print("15. CREATING PROMPTS")
print("=" * 70)


def make_prompt(func1, func2):

    return (
        "Determine if Function 1 and Function 2 are "
        "semantically equivalent.\n"
        "### Function 1:\n"
        f"{func1}\n"
        "### Function 2:\n"
        f"{func2}\n"
        "### Equivalent:"
    )


print("Creating training prompts...")

sample_train_df["prompt"] = [
    make_prompt(a, b)
    for a, b in zip(
        sample_train_df["func1"],
        sample_train_df["func2"]
    )
]


print("Creating validation prompts...")

val_df["prompt"] = [
    make_prompt(a, b)
    for a, b in zip(
        val_df["func1"],
        val_df["func2"]
    )
]


# ============================================================
# 17. FINAL COLUMN STRUCTURE
# ============================================================

sample_train_df = sample_train_df[
    [
        "id",
        "func1",
        "func2",
        "prompt",
        "label"
    ]
]

val_df = val_df[
    [
        "id",
        "func1",
        "func2",
        "prompt",
        "label"
    ]
]


# ============================================================
# 18. FINAL SHUFFLE
# ============================================================

sample_train_df = sample_train_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(
    drop=True
)

val_df = val_df.reset_index(
    drop=True
)


# ============================================================
# 19. CONVERT TO HF DATASETS
# ============================================================

print("\n" + "=" * 70)
print("16. CONVERTING TO HUGGING FACE DATASETS")
print("=" * 70)

hf_train = Dataset.from_pandas(
    sample_train_df,
    preserve_index=False
)

hf_val = Dataset.from_pandas(
    val_df,
    preserve_index=False
)


dataset = DatasetDict({
    "train": hf_train,
    "validation": hf_val
})


print("\nDataset structure:")

print(dataset)


# ============================================================
# 20. FINAL SANITY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("17. FINAL SANITY CHECKS")
print("=" * 70)


# Correct train size

assert len(dataset["train"]) == TRAIN_SAMPLE_SIZE


# Correct validation size

assert len(dataset["validation"]) == len(val_df)


# Labels

assert set(
    dataset["train"]["label"]
).issubset({0, 1})

assert set(
    dataset["validation"]["label"]
).issubset({0, 1})


# IDs unique

assert (
    len(set(dataset["train"]["id"]))
    ==
    len(dataset["train"])
)

assert (
    len(set(dataset["validation"]["id"]))
    ==
    len(dataset["validation"])
)


# Prompt doesn't contain explicit label

assert all(
    not str(label) in prompt[-5:]
    for prompt, label in zip(
        dataset["train"]["prompt"],
        dataset["train"]["label"]
    )
)


print("Training rows:", len(dataset["train"]))
print("Validation rows:", len(dataset["validation"]))

print("Training IDs unique: YES")
print("Validation IDs unique: YES")
print("Labels valid: YES")
print("No symmetry augmentation: YES")
print("No train/validation leakage: YES")


# ============================================================
# 21. SAVE DATASET
# ============================================================

print("\n" + "=" * 70)
print("18. SAVING DATASET")
print("=" * 70)

os.makedirs(
    OUTPUT,
    exist_ok=True
)

dataset.save_to_disk(
    OUTPUT
)

print(
    f"Dataset saved to: "
    f"{OUTPUT}"
)


# ============================================================
# 22. SAVE AUDIT REPORT
# ============================================================

audit = {

    "input_file": INPUT,

    "raw_rows": int(len(pd.read_json(INPUT, lines=True))),

    "clean_unique_pairs": int(len(df)),

    "full_training_pool": int(len(train_df)),

    "training_sample_size": int(
        len(sample_train_df)
    ),

    "validation_size": int(
        len(val_df)
    ),

    "symmetry_augmentation": False,

    "validation_fraction": VAL_FRACTION,

    "random_state": RANDOM_STATE,

    "label_distribution_train": {
        str(k): int(v)
        for k, v in
        sample_train_df["label"]
        .value_counts()
        .to_dict()
        .items()
    },

    "label_distribution_validation": {
        str(k): int(v)
        for k, v in
        val_df["label"]
        .value_counts()
        .to_dict()
        .items()
    },

    "train_validation_id_overlap": len(
        sample_val_overlap
    ),

    "max_seq_length_for_training": 1024,

    "notes": (
        "Training uses original unique pairs only. "
        "Symmetric pairs are not included in training. "
        "Symmetry will be handled during inference."
    )
}


report_path = os.path.join(
    OUTPUT,
    "preprocessing_report.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        audit,
        f,
        indent=2
    )


print(
    f"Audit report saved to: "
    f"{report_path}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(
    f"Training:   {len(sample_train_df):,}"
)

print(
    f"Validation: {len(val_df):,}"
)

print(
    f"Output:     {OUTPUT}"
)

print("=" * 70)