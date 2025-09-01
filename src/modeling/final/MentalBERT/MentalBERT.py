import pandas as pd
import numpy as np
import torch
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset

# ===== ROOT & PATHS =====
# Project root
ROOT = Path(__file__).resolve().parents[4]

# Paths to data splits
SPLITS_DIR   = ROOT / "data" / "processed" / "splits"
TRAINVAL_CSV = SPLITS_DIR / "train_val.csv"
TEST_CSV     = SPLITS_DIR / "test.csv"

# Output folders/files for this model
FINAL_DIR = ROOT / "modeling" / "final" / "MentalBERT"
MODEL_DIR = FINAL_DIR / "mentalbert_final_model"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_CSV = FINAL_DIR / "mentalbert_test_predictions.csv"
METRICS_CSV     = FINAL_DIR / "mentalbert_final_metrics.csv"
PARAMS_CSV      = FINAL_DIR / "mentalbert_used_hyperparameters.csv"

# ===== Model / Device =====
# Use the MentalBERT checkpoint; choose GPU if available
model_name = "mental/mental-bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ===== Load Data =====
# Train+validation split and held-out test set
train_df = pd.read_csv(TRAINVAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

# ===== Tokenizer & Datasets =====
# AutoTokenizer loads the correct tokenizer for the checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    """
    Tokenize a batch with padding/truncation up to 256 tokens,
    and attach labels from the 'Lable' column for supervised training.
    """
    tokens = tokenizer(batch['Text'], padding=True, truncation=True, max_length=256)
    tokens["labels"] = batch["Lable"]
    return tokens

# Convert pandas DataFrames to HF Datasets and apply tokenization
train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
test_dataset  = Dataset.from_pandas(test_df).map(tokenize, batched=True)

# ===== Best Params (from prior optimization) =====
best_params = {
    "learning_rate": 4.969782981573154e-05,
    "batch_size": 8,
    "epochs": 3
}

# ===== Model & Training Args =====
# Load the sequence classification head with 2 labels (binary task)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),                         # where Trainer writes artifacts
    num_train_epochs=best_params["epochs"],            # training epochs
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=best_params["batch_size"],
    learning_rate=best_params["learning_rate"],
    save_strategy="no",                                # don't keep intermediate checkpoints
    report_to="none"                                   # no external logging
)

# ===== Metrics =====
def compute_metrics(pred):
    """
    Compute standard binary classification metrics:
    accuracy, precision, recall, and F1 (positive class as positive label).
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# ===== Train =====
# Hugging Face Trainer handles training loop, eval, and metric computation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

start = time.time()
trainer.train()
runtime = time.time() - start

# ===== Save Model =====
# Persist the fine-tuned model and tokenizer for later inference
trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

# ===== Predict & Save =====
# Run inference on the test set and store predicted labels alongside the test data
preds = trainer.predict(test_dataset)
pred_labels = np.argmax(preds.predictions, axis=1)
test_df["prediction"] = pred_labels
test_df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8")

# Save evaluation metrics and runtime
metrics = compute_metrics(preds)
metrics["runtime_seconds"] = runtime
pd.DataFrame([metrics]).to_csv(METRICS_CSV, index=False)

# Save the hyperparameters actually used (handy for reproducibility)
param_log = best_params.copy()
param_log["model"] = model_name
pd.DataFrame([param_log]).to_csv(PARAMS_CSV, index=False)

print("Done! Saved to:", FINAL_DIR)
