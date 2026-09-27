import os
import numpy as np
import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_ID = "HuggingFaceTB/SmolLM-1.7B"

DATASET_PATH = "processed_codeclone_dataset"

OUTPUT_DIR = "smollm_classifier_qlora"

MAX_SEQ_LENGTH = 1536

SEED = 42


# ============================================================
# 2. DEVICE / PRECISION CHECK
# ============================================================

print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU:", torch.cuda.get_device_name(0))

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )

    print(
        "BF16 supported:",
        torch.cuda.is_bf16_supported()
    )


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

dataset = load_from_disk(
    DATASET_PATH
)

print(dataset)

print(
    "Train:",
    len(dataset["train"])
)

print(
    "Validation:",
    len(dataset["validation"])
)


# ============================================================
# 4. LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID
)

# SmolLM does not have a dedicated pad token.
if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print(
    "Pad token:",
    repr(tokenizer.pad_token)
)

print(
    "Pad token ID:",
    tokenizer.pad_token_id
)


# ============================================================
# 5. TOKENIZATION
# ============================================================

print("\n" + "=" * 70)
print("TOKENIZING")
print("=" * 70)

def tokenize_function(examples):
    return tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH
    )

# Target columns to drop if they exist
columns_to_drop = ["func1", "func2", "prompt", "id", "orientation"]

# Map split-by-split to avoid missing column errors
tokenized_datasets = dataset.copy()

for split in dataset.keys():
    # Only remove columns that actually exist in the current split
    valid_removals = [col for col in columns_to_drop if col in dataset[split].column_names]
    
    tokenized_datasets[split] = dataset[split].map(
        tokenize_function,
        batched=True,
        remove_columns=valid_removals,
        desc=f"Tokenizing {split}"
    )

print(tokenized_datasets)


# ============================================================
# 6. 4-BIT QUANTIZATION
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURING 4-BIT QLoRA")
print("=" * 70)

compute_dtype = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

print(
    "4-bit compute dtype:",
    compute_dtype
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 7. LOAD CLASSIFICATION MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING SmolLM-1.7B")
print("=" * 70)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    quantization_config=bnb_config,
    device_map="auto",
    # Safer than requiring Flash Attention 2.
    attn_implementation="sdpa",
)

model.config.pad_token_id = tokenizer.pad_token_id

model.config.use_cache = False

print(
    "Model loaded."
)


# ============================================================
# 8. PREPARE FOR K-BIT TRAINING
# ============================================================

print("\n" + "=" * 70)
print("PREPARING QLoRA")
print("=" * 70)

model = prepare_model_for_kbit_training(
    model
)


# ============================================================
# 9. LoRA CONFIGURATION
# ============================================================

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="SEQ_CLS",

    # Critical:
    # train the newly initialized classification head.
    modules_to_save=["score"],
)


model = get_peft_model(
    model,
    peft_config
)

model.print_trainable_parameters()


# ============================================================
# 10. DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True
)


# ============================================================
# 11. TRAINING ARGUMENTS
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURING TRAINING")
print("=" * 70)

use_bf16 = torch.cuda.is_bf16_supported()

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,
    max_steps = 100,
    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    num_train_epochs=1,

    # Safer for 8 GB VRAM.
    per_device_train_batch_size=4,

    # Effective batch = 1 * 16 = 16.
    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    lr_scheduler_type="cosine",

    warmup_steps=1770,

    weight_decay=0.01,

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    optim="paged_adamw_8bit",

    bf16=use_bf16,

    fp16=not use_bf16,

    gradient_checkpointing=False,
    group_by_length = True,

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    # We will do one dedicated validation prediction
    # after training rather than repeatedly evaluating
    # 25K examples during the 59K-step training run.

    eval_strategy="no",

    # --------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------

    save_strategy="no",

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_steps=100,

    report_to="none",

    seed=SEED,

    data_seed=SEED,

    # Prevent unnecessary reporting overhead.
    disable_tqdm=False,
)

print("=" * 60)
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3,
        "GB"
    )

print(
    "Model device:",
    next(model.parameters()).device
)

print(
    "Allocated:",
    torch.cuda.memory_allocated() / 1024**3,
    "GB"
)

print(
    "Reserved:",
    torch.cuda.memory_reserved() / 1024**3,
    "GB"
)

print("=" * 60)


# ============================================================
# 12. TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_datasets["train"],

    # No repeated evaluation during training.
    eval_dataset=None,

    data_collator=data_collator,

    processing_class=tokenizer,
)


# ============================================================
# 13. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

trainer.train()


# ============================================================
# 14. SAVE ADAPTER
# ============================================================

FINAL_DIR = (
    OUTPUT_DIR +
    "_final"
)

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

trainer.save_model(
    FINAL_DIR
)

tokenizer.save_pretrained(
    FINAL_DIR
)

print(
    f"Saved to: {FINAL_DIR}"
)


# ============================================================
# 15. FINAL VALIDATION PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING FINAL VALIDATION")
print("=" * 70)


validation_output = trainer.predict(
    tokenized_datasets["validation"]
)

logits = validation_output.predictions

labels = validation_output.label_ids


# ------------------------------------------------------------
# Convert logits -> probabilities
# ------------------------------------------------------------

logits_tensor = torch.tensor(
    logits
)

probabilities = torch.softmax(
    logits_tensor,
    dim=-1
).numpy()

p_class_1 = probabilities[:, 1]


# ============================================================
# 16. SAVE VALIDATION OUTPUTS
# ============================================================

np.save(
    "classifier_val_probs.npy",
    p_class_1
)

np.save(
    "classifier_val_logits.npy",
    logits
)

np.save(
    "classifier_val_labels.npy",
    labels
)

print(
    "Saved:"
)

print(
    "  classifier_val_probs.npy"
)

print(
    "  classifier_val_logits.npy"
)

print(
    "  classifier_val_labels.npy"
)


# ============================================================
# 17. QUICK 0.5 BASELINE
# ============================================================

predictions_05 = (
    p_class_1 >= 0.5
).astype(int)

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

print("\n" + "=" * 70)
print("QUICK VALIDATION RESULT @ THRESHOLD 0.5")
print("=" * 70)

print(
    "F1:",
    f1_score(
        labels,
        predictions_05
    )
)

print(
    "Precision:",
    precision_score(
        labels,
        predictions_05
    )
)

print(
    "Recall:",
    recall_score(
        labels,
        predictions_05
    )
)


# ============================================================
# 18. THRESHOLD SEARCH
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION THRESHOLD SEARCH")
print("=" * 70)

best_threshold = 0.5
best_f1 = 0.0

for threshold in np.arange(
    0.05,
    0.951,
    0.005
):

    predictions = (
        p_class_1 >= threshold
    ).astype(int)

    score = f1_score(
        labels,
        predictions
    )

    if score > best_f1:

        best_f1 = score
        best_threshold = threshold


print(
    f"Best threshold: "
    f"{best_threshold:.3f}"
)

print(
    f"Best validation F1: "
    f"{best_f1:.6f}"
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFIER TRAINING COMPLETE")
print("=" * 70)