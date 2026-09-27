from datasets import load_from_disk
from transformers import AutoTokenizer
import numpy as np

MODEL_ID = "HuggingFaceTB/SmolLM-1.7B"
DATASET_PATH = "processed_codeclone_dataset"

N = 10000

print("Loading dataset...")
dataset = load_from_disk(DATASET_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("\nChecking label tokens...")

for text in ["0", "1", " 0", " 1"]:
    ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    print(
        repr(text),
        "=>",
        ids,
        "length:",
        len(ids)
    )


print("\nSampling prompts...")

sample = dataset["train"].shuffle(
    seed=42
).select(
    range(min(N, len(dataset["train"])))
)

lengths = []

for prompt in sample["prompt"]:

    ids = tokenizer(
        prompt,
        truncation=False,
        add_special_tokens=True
    )["input_ids"]

    lengths.append(len(ids))


lengths = np.array(lengths)

print("\n" + "=" * 60)
print("TOKEN LENGTH AUDIT")
print("=" * 60)

for percentile in [
    50,
    75,
    90,
    95,
    97,
    98,
    99,
    99.5,
    99.9
]:

    print(
        f"P{percentile:<5}: "
        f"{np.percentile(lengths, percentile):.0f}"
    )

print(
    f"Maximum: {lengths.max()}"
)

print(
    f"Mean:    {lengths.mean():.1f}"
)

print(
    f"Median:  {np.median(lengths):.0f}"
)

print("\nTruncation estimates:")

for limit in [768, 1024, 1280, 1536, 1792, 2048]:

    pct = (
        100 *
        np.mean(lengths > limit)
    )

    print(
        f"{limit:4d} tokens -> "
        f"{pct:.2f}% truncated"
    )