import json
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict


# ============================================================
# CONFIGURATION
# ============================================================

INPUT = "train_small.jsonl"
OUTPUT = "processed_codeclone_dataset"

SEED = 42
VAL_SIZE = 0.05


# ============================================================
# 1. LOAD JSONL
# ============================================================

print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

print(f"Input file: {INPUT}")

df = pd.read_json(INPUT, lines=True)

raw_rows = len(df)

print(f"Raw rows: {raw_rows:,}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 2. SCHEMA VALIDATION
# ============================================================

required = {
    "id",
    "func1",
    "func2",
    "label"
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("\nRequired columns found:")
print("  id")
print("  func1")
print("  func2")
print("  label")


# ============================================================
# 3. ID AUDIT
# ============================================================

print("\n" + "=" * 70)
print("2. ID AUDIT")
print("=" * 70)

duplicate_id_count = int(
    df["id"].duplicated().sum()
)

unique_id_count = int(
    df["id"].nunique()
)

print(f"Unique IDs:    {unique_id_count:,}")
print(f"Duplicate IDs: {duplicate_id_count:,}")

if duplicate_id_count > 0:
    print(
        "\nWARNING:"
        " Duplicate IDs exist in the raw dataset."
        "\nWe will NOT automatically remove them."
        "\nPair-level deduplication below is the actual"
        "\ndeduplication criterion."
    )


# ============================================================
# 4. BASIC VALIDATION / CLEANING
# ============================================================

print("\n" + "=" * 70)
print("3. BASIC VALIDATION")
print("=" * 70)

before = len(df)

df = df.dropna(
    subset=[
        "id",
        "func1",
        "func2",
        "label"
    ]
).copy()

removed_missing = before - len(df)

print(
    f"Removed rows with missing values: "
    f"{removed_missing:,}"
)


# ------------------------------------------------------------
# Convert labels to integers
# ------------------------------------------------------------

try:

    df["label"] = df["label"].astype(int)

except Exception as e:

    raise ValueError(
        f"Could not convert labels to integers: {e}"
    )


# ------------------------------------------------------------
# Keep only valid binary labels
# ------------------------------------------------------------

before = len(df)

df = df[
    df["label"].isin([0, 1])
].copy()

removed_invalid_labels = before - len(df)

print(
    f"Removed rows with invalid labels: "
    f"{removed_invalid_labels:,}"
)


# ============================================================
# 5. SAFE CODE NORMALIZATION
# ============================================================

print("\n" + "=" * 70)
print("4. CODE NORMALIZATION")
print("=" * 70)


def normalize_code(code):
    """
    Conservative code normalization.

    Changes:
        CRLF -> LF
        CR   -> LF
        trailing whitespace removed per line
        surrounding whitespace removed

    Does NOT:
        rename variables
        remove comments
        alter indentation
        alter literals
        reformat syntax
        alter program logic
    """

    if not isinstance(code, str):
        code = str(code)

    # Normalize line endings
    code = code.replace("\r\n", "\n")
    code = code.replace("\r", "\n")

    # Remove trailing whitespace from each line.
    # Leading whitespace/indentation is preserved.
    code = "\n".join(
        line.rstrip()
        for line in code.split("\n")
    )

    # Remove leading/trailing blank whitespace
    return code.strip()


df["func1"] = df["func1"].map(normalize_code)
df["func2"] = df["func2"].map(normalize_code)

print("Normalization complete.")


# ============================================================
# 6. REMOVE EMPTY FUNCTIONS
# ============================================================

print("\n" + "=" * 70)
print("5. EMPTY FUNCTION CHECK")
print("=" * 70)

before = len(df)

df = df[
    (df["func1"].str.len() > 0) &
    (df["func2"].str.len() > 0)
].copy()

removed_empty = before - len(df)

print(
    f"Removed empty-function rows: "
    f"{removed_empty:,}"
)


# ============================================================
# 7. CREATE ORDER-INDEPENDENT PAIR KEY
# ============================================================

print("\n" + "=" * 70)
print("6. PAIR-LEVEL DEDUPLICATION KEY")
print("=" * 70)


def make_pair_key(func1, func2):
    """
    Creates an order-independent SHA256 key.

    Therefore:

        (A, B) == (B, A)

    The null byte separator prevents ambiguous
    concatenations.
    """

    if func1 <= func2:
        first = func1
        second = func2
    else:
        first = func2
        second = func1

    raw = (
        first.encode("utf-8")
        + b"\x00"
        + second.encode("utf-8")
    )

    return hashlib.sha256(raw).hexdigest()


df["pair_key"] = [
    make_pair_key(a, b)
    for a, b in zip(
        df["func1"],
        df["func2"]
    )
]

print("Pair keys created.")


# ============================================================
# 8. DETECT CONTRADICTORY LABELS
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

num_conflicting_pairs = len(
    conflicting_keys
)

print(
    f"Conflicting unordered pairs: "
    f"{num_conflicting_pairs:,}"
)

if num_conflicting_pairs > 0:

    print(
        "\nWARNING:"
        "\nThe same unordered function pair occurs "
        "with both label 0 and label 1."
    )

    print(
        "\nRemoving ALL rows belonging to "
        "those contradictory pairs."
    )

    before = len(df)

    df = df[
        ~df["pair_key"].isin(conflicting_keys)
    ].copy()

    print(
        f"Rows removed due to conflicts: "
        f"{before - len(df):,}"
    )

else:

    print(
        "No contradictory unordered pairs found."
    )


# ============================================================
# 9. DEDUPLICATE UNORDERED PAIRS
# ============================================================

print("\n" + "=" * 70)
print("8. EXACT / REVERSED PAIR DEDUPLICATION")
print("=" * 70)

before = len(df)

df = df.drop_duplicates(
    subset=["pair_key"],
    keep="first"
).copy()

removed_duplicates = before - len(df)

print(
    f"Removed duplicate/reversed pairs: "
    f"{removed_duplicates:,}"
)

print(
    f"Unique unordered pairs remaining: "
    f"{len(df):,}"
)

# Pair key no longer needed
df = df.drop(
    columns=["pair_key"]
)


# ============================================================
# 10. LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("9. LABEL DISTRIBUTION")
print("=" * 70)

label_counts = (
    df["label"]
    .value_counts()
    .sort_index()
)

label_percentages = (
    df["label"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

for label in [0, 1]:

    count = int(
        label_counts.get(label, 0)
    )

    percentage = float(
        label_percentages.get(label, 0)
    )

    print(
        f"Label {label}: "
        f"{count:,} "
        f"({percentage:.2f}%)"
    )


clean_rows = len(df)


# ============================================================
# 11. STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 70)
print("10. TRAIN / VALIDATION SPLIT")
print("=" * 70)

print(
    f"Validation fraction: {VAL_SIZE:.2%}"
)

train_df, val_df = train_test_split(
    df,
    test_size=VAL_SIZE,
    stratify=df["label"],
    random_state=SEED
)

train_df = train_df.reset_index(
    drop=True
)

val_df = val_df.reset_index(
    drop=True
)

train_rows_before_symmetry = len(
    train_df
)

validation_rows = len(
    val_df
)

print(
    f"Train before symmetry: "
    f"{train_rows_before_symmetry:,}"
)

print(
    f"Validation: "
    f"{validation_rows:,}"
)


# ============================================================
# 12. MARK ORIGINAL TRAINING ORIENTATION
# ============================================================

print("\n" + "=" * 70)
print("11. SYMMETRIC AUGMENTATION")
print("=" * 70)

# Original examples
train_df["orientation"] = "original"


# ============================================================
# 13. CREATE REVERSED TRAINING EXAMPLES
# ============================================================

train_sym = train_df.copy()

# Swap Function 1 and Function 2
tmp = train_sym["func1"].copy()

train_sym["func1"] = train_sym["func2"]

train_sym["func2"] = tmp

# Mark them as reversed
train_sym["orientation"] = "reversed"

# IMPORTANT:
#
# id stays the same
# label stays the same
#
# because:
#
# semantic_equivalence(A, B)
# =
# semantic_equivalence(B, A)


# ============================================================
# 14. COMBINE ORIGINAL + REVERSED
# ============================================================

train_df = pd.concat(
    [
        train_df,
        train_sym
    ],
    ignore_index=True
)

train_rows_after_symmetry = len(
    train_df
)

print(
    f"Original training rows: "
    f"{train_rows_before_symmetry:,}"
)

print(
    f"Symmetric training rows: "
    f"{train_rows_before_symmetry:,}"
)

print(
    f"Final augmented training rows: "
    f"{train_rows_after_symmetry:,}"
)


# ============================================================
# 15. SHUFFLE AUGMENTED TRAINING DATA
# ============================================================

print("\nShuffling augmented training data...")

train_df = train_df.sample(
    frac=1.0,
    random_state=SEED
).reset_index(
    drop=True
)

print("Training data shuffled.")


# ============================================================
# 16. COMMON PROMPT
# ============================================================

print("\n" + "=" * 70)
print("12. CREATING COMMON PROMPTS")
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


# ------------------------------------------------------------
# Training prompts
# ------------------------------------------------------------

train_df["prompt"] = [
    make_prompt(func1, func2)
    for func1, func2 in zip(
        train_df["func1"],
        train_df["func2"]
    )
]


# ------------------------------------------------------------
# Validation prompts
# ------------------------------------------------------------

val_df["prompt"] = [
    make_prompt(func1, func2)
    for func1, func2 in zip(
        val_df["func1"],
        val_df["func2"]
    )
]

print("Prompts created.")


# ============================================================
# 17. FINAL DATASET COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("13. FINAL DATASET STRUCTURE")
print("=" * 70)

# Training:
#
# id
# func1
# func2
# prompt
# label
# orientation
#
# Validation:
#
# id
# func1
# func2
# prompt
# label
#
# NOTE:
# No label is embedded inside prompt.
#
# The generative SFT training script will append
# the label to the prompt during training.
#
# Validation/test inference will use the prompt
# WITHOUT the label.

train_df = train_df[
    [
        "id",
        "func1",
        "func2",
        "prompt",
        "label",
        "orientation"
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
# 18. FINAL SANITY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("14. SANITY CHECKS")
print("=" * 70)

# ------------------------------------------------------------
# Check train orientation counts
# ------------------------------------------------------------

print("\nTraining orientations:")

print(
    train_df["orientation"]
    .value_counts()
)


# ------------------------------------------------------------
# Check labels
# ------------------------------------------------------------

assert set(
    train_df["label"].unique()
).issubset({0, 1})

assert set(
    val_df["label"].unique()
).issubset({0, 1})

print(
    "\nLabels contain only 0 and 1."
)


# ------------------------------------------------------------
# Check no label appears inside prompt
# ------------------------------------------------------------

# We don't perform a literal "0"/"1" check because
# the source code itself may naturally contain digits.
#
# Instead, verify that prompt generation itself never
# explicitly appends the label.
#
# This is guaranteed by make_prompt() above.

print(
    "Prompt does not explicitly append the label."
)


# ------------------------------------------------------------
# Check train/validation ID overlap
# ------------------------------------------------------------

train_ids = set(
    train_df["id"].tolist()
)

val_ids = set(
    val_df["id"].tolist()
)

overlap = train_ids.intersection(
    val_ids
)

print(
    f"\nTrain/validation ID overlap: "
    f"{len(overlap):,}"
)

if len(overlap) > 0:

    print(
        "WARNING: Some IDs occur in both "
        "train and validation."
    )

else:

    print(
        "No train/validation ID overlap."
    )


# ------------------------------------------------------------
# Check that every training ID has two orientations
# ------------------------------------------------------------

orientation_counts = (
    train_df
    .groupby("id")["orientation"]
    .nunique()
)

bad_orientation_ids = (
    orientation_counts[
        orientation_counts != 2
    ]
)

print(
    f"\nTraining IDs without exactly "
    f"two orientations: "
    f"{len(bad_orientation_ids):,}"
)

if len(bad_orientation_ids) > 0:

    print(
        "WARNING: Some training IDs do not "
        "have both original and reversed examples."
    )

else:

    print(
        "Every training ID has both orientations."
    )


# ============================================================
# 19. CONVERT TO HUGGING FACE DATASETS
# ============================================================

print("\n" + "=" * 70)
print("15. CONVERTING TO HUGGING FACE DATASETS")
print("=" * 70)

hf_train = Dataset.from_pandas(
    train_df,
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
# 20. SAVE DATASET
# ============================================================

print("\n" + "=" * 70)
print("16. SAVING DATASET")
print("=" * 70)

dataset.save_to_disk(
    OUTPUT
)

print(
    f"Dataset saved to: {OUTPUT}"
)


# ============================================================
# 21. SAVE PREPROCESSING AUDIT REPORT
# ============================================================

print("\n" + "=" * 70)
print("17. SAVING AUDIT REPORT")
print("=" * 70)

output_path = Path(OUTPUT)

report = {
    "input": INPUT,
    "seed": SEED,
    "validation_fraction": VAL_SIZE,

    "raw_rows": int(raw_rows),

    "removed_missing_rows": int(
        removed_missing
    ),

    "removed_invalid_label_rows": int(
        removed_invalid_labels
    ),

    "removed_empty_rows": int(
        removed_empty
    ),

    "conflicting_unordered_pairs": int(
        num_conflicting_pairs
    ),

    "removed_duplicate_or_reversed_pairs": int(
        removed_duplicates
    ),

    "clean_unique_pairs": int(
        clean_rows
    ),

    "train_rows_before_symmetry": int(
        train_rows_before_symmetry
    ),

    "train_rows_after_symmetry": int(
        train_rows_after_symmetry
    ),

    "validation_rows": int(
        validation_rows
    ),

    "raw_unique_ids": int(
        unique_id_count
    ),

    "raw_duplicate_ids": int(
        duplicate_id_count
    ),

    "train_validation_id_overlap": int(
        len(overlap)
    ),

    "label_distribution": {
        str(label): int(count)
        for label, count
        in df["label"]
        .value_counts()
        .sort_index()
        .items()
    }
}


with open(
    output_path / "preprocessing_report.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


print(
    "Audit report saved to:"
)

print(
    output_path /
    "preprocessing_report.json"
)


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(
    f"Raw rows:                    {raw_rows:,}"
)

print(
    f"Clean unique pairs:          {clean_rows:,}"
)

print(
    f"Training before symmetry:    "
    f"{train_rows_before_symmetry:,}"
)

print(
    f"Training after symmetry:     "
    f"{train_rows_after_symmetry:,}"
)

print(
    f"Validation:                  "
    f"{validation_rows:,}"
)

print(
    f"Output:                      {OUTPUT}"
)

print("=" * 70)