import os
import time
import numpy as np
import torch

from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)

from peft import (
    LoraConfig,
    get_peft_model
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_ID = "meta-llama/Llama-3.2-1B"
DATASET_PATH = "processed_codeclone_5k"
OUTPUT_DIR = "llama_classifier_bf16_lora"
MAX_SEQ_LENGTH = 1024
SEED = 42

# Training
NUM_EPOCHS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 470

# LoRA
LORA_R = 16
LORA_ALPHA = 64
LORA_DROPOUT = 0.05


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

set_seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. SYSTEM INFORMATION
# ============================================================

print("=" * 70)
print("SYSTEM INFORMATION")
print("=" * 70)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Do not start training on CPU.")

print(f"GPU: {torch.cuda.get_device_name(0)}")
gpu_properties = torch.cuda.get_device_properties(0)
print(f"GPU VRAM: {gpu_properties.total_memory / 1024**3:.2f} GB")


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

dataset = load_from_disk(DATASET_PATH)
print(dataset)
print(f"Training examples: {len(dataset['train']):,}")
print(f"Validation examples: {len(dataset['validation']):,}")


# ============================================================
# 5. LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# Llama normally has no dedicated padding token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Pad token was not present.")
    print(f"Using EOS token as pad token: {tokenizer.pad_token}")

tokenizer.padding_side = "right"

print(f"Tokenizer vocab size: {len(tokenizer):,}")
print(f"PAD token: {tokenizer.pad_token} ({tokenizer.pad_token_id})")
print(f"EOS token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")


# ============================================================
# 6. TOKENIZATION
# ============================================================

print("\n" + "=" * 70)
print("TOKENIZING DATA")
print("=" * 70)
print(f"Maximum sequence length: {MAX_SEQ_LENGTH}")

def tokenize_function(examples):
    return tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=["id", "func1", "func2", "prompt"],
    desc="Tokenizing"
)

print("\nTokenization complete.")


# ============================================================
# 7. DATA COLLATOR
# ============================================================

print("\n" + "=" * 70)
print("CREATING DATA COLLATOR")
print("=" * 70)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8
)
print("Dynamic padding enabled.")


# ============================================================
# 8. LOAD LLAMA CLASSIFICATION MODEL (PURE BF16)
# ============================================================

print("\n" + "=" * 70)
print("LOADING LLAMA 3.2 1B (PURE BF16)")
print("=" * 70)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
).to("cuda")

model.config.pad_token_id = tokenizer.pad_token_id
model.config.num_labels = 2
model.config.use_cache = False

print(f"Model pad_token_id: {model.config.pad_token_id}")
print(f"Number of labels: {model.config.num_labels}")


# ============================================================
# 9. LORA CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("APPLYING LORA")
print("=" * 70)

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="SEQ_CLS",
    modules_to_save=["score"] # Keeps the classification head trainable
)

model = get_peft_model(model, peft_config)
print("\nTrainable parameters:")
model.print_trainable_parameters()


# ============================================================
# 10. VERIFY MODEL DEVICE
# ============================================================

print("\n" + "=" * 70)
print("MODEL DEVICE CHECK")
print("=" * 70)

first_parameter = next(model.parameters())
print(f"First parameter device: {first_parameter.device}")
print(f"First parameter dtype: {first_parameter.dtype}")


# ============================================================
# 11. METRICS
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# ============================================================
# 12. TRAINING ARGUMENTS
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURING TRAINING")
print("=" * 70)

print(f"Physical batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Effective batch size: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

use_bf16 = torch.cuda.is_bf16_supported()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # TRAINING
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    
    # OPTIMIZATION
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    optim="adamw_torch_fused",
    
    # PRECISION
    bf16=use_bf16,
    fp16=not use_bf16,
    gradient_checkpointing=False,
    
    # EVALUATION / SAVING
    eval_strategy="no",
    save_strategy="no",
    
    # LOGGING
    logging_steps=50,
    logging_first_step=True,
    report_to="none",
    
    # DATALOADER
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    
    # REPRODUCIBILITY
    seed=SEED,
    data_seed=SEED,
    
    # MISC
    remove_unused_columns=True
)


# ============================================================
# 13. CREATE TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# ============================================================
# 14. GPU MEMORY BEFORE TRAINING
# ============================================================

print("\n" + "=" * 70)
print("GPU MEMORY BEFORE TRAINING")
print("=" * 70)

torch.cuda.empty_cache()
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


# ============================================================
# 15. TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING LLAMA TRAINING")
print("=" * 70)
print("Starting now...")

start_time = time.time()
train_result = trainer.train()
elapsed = time.time() - start_time


# ============================================================
# 16. TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Training time: {elapsed / 3600:.2f} hours")
print(f"Training time: {elapsed / 60:.1f} minutes")
print(f"Training metrics:\n{train_result.metrics}")


# ============================================================
# 17. SAVE TRAINED ADAPTER
# ============================================================

FINAL_OUTPUT = OUTPUT_DIR + "_final"

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

trainer.save_model(FINAL_OUTPUT)
tokenizer.save_pretrained(FINAL_OUTPUT)
print(f"Adapter saved to: {FINAL_OUTPUT}")


# ============================================================
# 18. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("RUNNING FINAL VALIDATION")
print("=" * 70)

eval_start = time.time()
metrics = trainer.evaluate(eval_dataset=tokenized_dataset["validation"])
eval_time = time.time() - eval_start

print(f"Validation time: {eval_time / 60:.1f} minutes")
print("\nValidation metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")


# ============================================================
# 19. SAVE VALIDATION LOGITS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING VALIDATION LOGITS")
print("=" * 70)

predictions = trainer.predict(tokenized_dataset["validation"])
logits = predictions.predictions
labels = predictions.label_ids

np.save(os.path.join(FINAL_OUTPUT, "validation_logits.npy"), logits)
np.save(os.path.join(FINAL_OUTPUT, "validation_labels.npy"), labels)


# ============================================================
# 20. EXTRACT CLASS-1 PROBABILITY
# ============================================================

logits_shifted = logits - logits.max(axis=1, keepdims=True)
exp_logits = np.exp(logits_shifted)
probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
p_class1 = probabilities[:, 1]

np.save(os.path.join(FINAL_OUTPUT, "validation_p_class1.npy"), p_class1)


# ============================================================
# 21. THRESHOLD SWEEP
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)

best_threshold = 0.5
best_f1 = 0.0
best_precision = 0.0
best_recall = 0.0

thresholds = np.arange(0.05, 0.951, 0.005)

for threshold in thresholds:
    predictions_binary = (p_class1 >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions_binary,
        average="binary",
        zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_precision = precision
        best_recall = recall

print(f"Best threshold: {best_threshold:.3f}")
print(f"Best validation F1: {best_f1:.6f}")
print(f"Precision: {best_precision:.6f}")
print(f"Recall: {best_recall:.6f}")


# ============================================================
# 22. SAVE THRESHOLD
# ============================================================

threshold_file = os.path.join(FINAL_OUTPUT, "best_threshold.txt")
with open(threshold_file, "w") as f:
    f.write(f"{best_threshold:.6f}\n")


# ============================================================
# 23. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("LLAMA CLASSIFIER PIPELINE COMPLETE")
print("=" * 70)
print(f"Model: {MODEL_ID}")
print(f"Training examples: {len(tokenized_dataset['train']):,}")
print(f"Validation examples: {len(tokenized_dataset['validation']):,}")
print(f"Max sequence length: {MAX_SEQ_LENGTH}")
print(f"Best validation F1: {best_f1:.6f}")
print(f"Best threshold: {best_threshold:.3f}")
print(f"Saved model: {FINAL_OUTPUT}")
print("=" * 70)