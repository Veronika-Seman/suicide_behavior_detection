import pandas as pd
import numpy as np
import torch
import time
from pathlib import Path
from transformers import XLNetTokenizerFast, XLNetForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset

# ===== ROOT & PATHS =====
# Resolve project root
ROOT = Path(__file__).resolve().parents[4]

# Input split files
SPLITS_DIR   = ROOT / "data" / "processed" / "splits"
TRAINVAL_CSV = SPLITS_DIR / "train_val.csv"
TEST_CSV     = SPLITS_DIR / "test.csv"

# Output directory for this model
FINAL_DIR = ROOT / "modeling" / "final" / "XLNet_Base"
MODEL_DIR = FINAL_DIR / "xlnet_final_model"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Output files
PREDICTIONS_CSV = FINAL_DIR / "xlnet_test_predictions.csv"
METRICS_CSV     = FINAL_DIR / "xlnet_final_metrics.csv"
PARAMS_CSV      = FINAL_DIR / "xlnet_used_hyperparameters.csv"

# ===== Model / Device =====
# Load XLNet base (cased) and pick CUDA if available
model_name = 'xlnet-base-cased'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ===== Load data =====
# Train+Validation and Test CSVs
train_val_df = pd.read_csv(TRAINVAL_CSV)
test_df      = pd.read_csv(TEST_CSV)

# ===== Tokenizer & Datasets =====
# Tokenize text inputs for XLNet
tokenizer = XLNetTokenizerFast.from_pretrained(model_name)

def tokenize(batch):
    """Tokenize text and attach labels from 'Lable' column."""
    tokens = tokenizer(batch['Text'], padding=True, truncation=True, max_length=256)
    tokens["labels"] = batch["Lable"]
    return tokens

# Convert DataFrames to HuggingFace Datasets and tokenize
train_dataset = Dataset.from_pandas(train_val_df).map(tokenize, batched=True)
test_dataset  = Dataset.from_pandas(test_df).map(tokenize, batched=True)

# ===== Best hyperparameters (from optimization) =====
best_params = {
    "learning_rate": 3.819059334176231e-05,
    "batch_size": 8,
    "epochs": 3
}

# ===== Model & Training Arguments =====
model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),                        # output folder for trainer artifacts
    num_train_epochs=best_params["epochs"],           # training epochs
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=best_params["batch_size"],
    learning_rate=best_params["learning_rate"],
    save_strategy="no",                               # skip checkpoint saving
    report_to="none"                                  # disable external logging
)

# ===== Metrics function =====
def compute_metrics(pred):
    """Compute Accuracy, Precision, Recall, and F1 (binary classification)."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# ===== Train model =====
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

# ===== Save model & tokenizer =====
trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

# ===== Predict & Save results =====
preds = trainer.predict(test_dataset)
pred_labels = np.argmax(preds.predictions, axis=1)

# Save test predictions alongside original data
test_df["prediction"] = pred_labels
test_df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8")

# Save evaluation metrics
metrics = compute_metrics(preds)
metrics["runtime_seconds"] = runtime
pd.DataFrame([metrics]).to_csv(METRICS_CSV, index=False)

# Save used hyperparameters for reproducibility
param_log = best_params.copy()
param_log["model"] = model_name
pd.DataFrame([param_log]).to_csv(PARAMS_CSV, index=False)

print("Done! Saved to:", FINAL_DIR)
