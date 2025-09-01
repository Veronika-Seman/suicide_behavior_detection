import pandas as pd
import numpy as np
import torch
import time
from pathlib import Path
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset


# --- Project paths ---
ROOT = Path(__file__).resolve().parents[4]

SPLITS_DIR = ROOT / "data" / "processed" / "splits"
FINAL_DIR  = ROOT / "src" / "modeling" / "final" / "BERT_Base"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# Input data
TRAIN_VAL_CSV = SPLITS_DIR / "train_val.csv"
TEST_CSV      = SPLITS_DIR / "test.csv"

# Output directories and files
MODEL_DIR     = FINAL_DIR / "bert_final_model"
PREDICTIONS   = FINAL_DIR / "bert_test_predictions.csv"
METRICS_CSV   = FINAL_DIR / "bert_final_metrics.csv"
PARAMS_CSV    = FINAL_DIR / "bert_used_hyperparameters.csv"


# --- Config ---
model_name = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")


# --- Data loading ---
train_val_df = pd.read_csv(TRAIN_VAL_CSV)
test_df      = pd.read_csv(TEST_CSV)

tokenizer = BertTokenizerFast.from_pretrained(model_name)

# Tokenization function
def tokenize(batch):
    tokens = tokenizer(batch["Text"], padding=True, truncation=True, max_length=256)
    tokens["labels"] = batch["Lable"]   # Ensure label column is mapped properly
    return tokens

# Convert pandas DataFrames → HuggingFace Datasets and tokenize
train_dataset = Dataset.from_pandas(train_val_df).map(tokenize, batched=True)
test_dataset  = Dataset.from_pandas(test_df).map(tokenize, batched=True)


# --- Best hyperparameters (from Optuna search) ---
best_params = {
    "learning_rate": 4.6812760476533354e-05,
    "batch_size": 8,
    "epochs": 3,
}


# --- Model setup ---
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),
    num_train_epochs=best_params["epochs"],
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=best_params["batch_size"],
    learning_rate=best_params["learning_rate"],
    save_strategy="no",    # Don't save intermediate checkpoints
    disable_tqdm=True,     # Clean console output
    report_to="none",      # Disable logging to external services
)

# Custom evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


# --- Trainer setup ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# --- Training ---
start = time.time()
trainer.train()
runtime = time.time() - start


# --- Save final model ---
trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))


# --- Predict on test set and save predictions ---
preds = trainer.predict(test_dataset)
pred_labels = np.argmax(preds.predictions, axis=1)
test_df["prediction"] = pred_labels
test_df.to_csv(PREDICTIONS, index=False, encoding="utf-8")


# --- Save metrics & hyperparameters ---
metrics = compute_metrics(preds)
metrics["runtime_seconds"] = runtime
pd.DataFrame([metrics]).to_csv(METRICS_CSV, index=False)

param_log = best_params.copy()
param_log["model"] = model_name
pd.DataFrame([param_log]).to_csv(PARAMS_CSV, index=False)


print("Done. All files saved to:", FINAL_DIR)
