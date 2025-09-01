import pandas as pd
import numpy as np
import torch
import time
from pathlib import Path
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset

# ===== Project root =====
# Resolve two levels up
ROOT = Path(__file__).resolve().parents[2]

# ===== Paths =====
SPLITS_DIR   = ROOT / "data" / "processed" / "splits"
TRAINVAL_CSV = SPLITS_DIR / "train_val.csv"
TEST_CSV     = SPLITS_DIR / "test.csv"

FINAL_DIR = ROOT / "src" / "modeling" / "final" / "RoBERTa"
MODEL_DIR = FINAL_DIR / "roberta_final_model"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_CSV = FINAL_DIR / "roberta_test_predictions.csv"
METRICS_CSV     = FINAL_DIR / "roberta_final_metrics.csv"
PARAMS_CSV      = FINAL_DIR / "roberta_used_hyperparameters.csv"

# ===== General setup =====
model_name = "roberta-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ===== Load data =====
train_val_df = pd.read_csv(TRAINVAL_CSV)
test_df      = pd.read_csv(TEST_CSV)

# ===== Tokenizer =====
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

def tokenize(batch):
    """
    Tokenize input text and attach labels from the 'Lable' column.
    """
    tokens = tokenizer(batch["Text"], padding=True, truncation=True, max_length=256)
    tokens["labels"] = batch["Lable"]
    return tokens

train_dataset = Dataset.from_pandas(train_val_df).map(tokenize, batched=True)
test_dataset  = Dataset.from_pandas(test_df).map(tokenize, batched=True)

# ===== Hyperparameters (from optimization) =====
best_params = {
    "learning_rate": 3.003179800231941e-05,
    "batch_size": 8,
    "epochs": 3,
}

# ===== Model =====
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# ===== Training arguments =====
training_args = TrainingArguments(
    output_dir=str(MODEL_DIR),
    num_train_epochs=best_params["epochs"],
    per_device_train_batch_size=best_params["batch_size"],
    per_device_eval_batch_size=best_params["batch_size"],
    learning_rate=best_params["learning_rate"],
    save_strategy="no",
    logging_dir=str(FINAL_DIR / "logs"),
    logging_steps=10,
    disable_tqdm=True,
    report_to="none",
)

# ===== Metrics =====
def compute_metrics(pred):
    """Compute Accuracy, Precision, Recall, F1 for binary classification."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# ===== Trainer =====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# ===== Train =====
start = time.time()
trainer.train()
runtime = time.time() - start

# ===== Save model & tokenizer =====
trainer.save_model(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

# ===== Predictions =====
preds = trainer.predict(test_dataset)
pred_labels = np.argmax(preds.predictions, axis=1)

test_df["prediction"] = pred_labels
test_df.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8")

# ===== Save metrics =====
metrics = compute_metrics(preds)
metrics["runtime_seconds"] = runtime
pd.DataFrame([metrics]).to_csv(METRICS_CSV, index=False)

# ===== Save used hyperparameters =====
param_log = best_params.copy()
param_log["model"] = model_name
pd.DataFrame([param_log]).to_csv(PARAMS_CSV, index=False)

print(" Done! All files saved to:", FINAL_DIR)
