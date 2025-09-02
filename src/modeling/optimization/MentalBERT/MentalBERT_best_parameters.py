import os
import csv
import numpy as np
import pandas as pd
import torch
import optuna
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from huggingface_hub import login
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# ==== Project Root & .env ====
ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".env", override=True)

# ==== Hugging Face Hub Login (from .env) ====
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
if hf_token:
    login(hf_token)
else:
    print("HUGGING_FACE_HUB_TOKEN not found in .env")

# ==== Paths ====
SPLITS_DIR     = ROOT / "data" / "processed" / "splits"
TRAINVAL_FIXED = SPLITS_DIR / "train_val_fixed.csv"
FOLDS_FIXED    = SPLITS_DIR / "fold_assignments_fixed.csv"

MENTAL_BERT_DIR   = ROOT / "modeling" / "optuna" / "MentalBERT"
OPTUNA_MODELS_DIR = MENTAL_BERT_DIR / "optuna_models"
BEST_MODEL_DIR    = MENTAL_BERT_DIR / "best_optuna_model"
RESULTS_FILE      = MENTAL_BERT_DIR / "optuna_trial_results.csv"
BEST_TRIAL_FILE   = MENTAL_BERT_DIR / "best_trial_summary.csv"

for p in [MENTAL_BERT_DIR, OPTUNA_MODELS_DIR, BEST_MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ==== General Settings ====
model_name = "mental/mental-bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ==== Load Data ====
assert TRAINVAL_FIXED.exists(), f"Missing: {TRAINVAL_FIXED}"
assert FOLDS_FIXED.exists(),    f"Missing: {FOLDS_FIXED}"

train_val_df = pd.read_csv(TRAINVAL_FIXED)
if "index" in train_val_df.columns:
    train_val_df.set_index("index", inplace=True)

folds_df = pd.read_csv(FOLDS_FIXED).astype({"index": int})

# ==== Tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch: dict) -> dict:
    """
     Prepares a batch for MentalBERT training:
   - Applies padding so all sequences in the batch have equal length.
   - Truncates sequences longer than 256 tokens.
   - Returns input_ids and attention_mask, and adds "labels" for supervised loss.
    """
    tokens = tokenizer(batch["Text"], padding=True, truncation=True, max_length=256)
    tokens["labels"] = batch["Lable"]
    return tokens

def compute_metrics(pred) -> dict:
    """
   Computes binary classification metrics:
  - Takes logits and true labels from Hugging Face EvalPrediction.
  - Uses argmax over logits to decide predicted class {0,1}.
  - Returns accuracy, precision, recall, and F1 (with class '1' as positive).
  - "eval_f1" is the main score used to compare trials and pick the best model.
    """
    labels = pred.label_ids
    preds  = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": p, "recall": r, "eval_f1": f1}


# Track global best across trials
best_f1 = 0.0
best_metrics = {}

def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna hyperparameter search.

    Workflow:
      1. Suggest learning_rate and batch_size.
      2. Perform 5-fold cross-validation using pre-defined folds.
      3. For each fold:
         - Split train/val sets by index.
         - Tokenize and wrap into HF Datasets.
         - Train DistilBERT for 3 epochs.
         - Evaluate on validation set.
         - Save model/tokenizer if best F1 so far.
      4. Compute average metrics across folds.
      5. Log trial results into optuna_trial_results.csv.
      6. Return average F1 for Optuna to maximize.
    """
    global best_f1, best_metrics

    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    fold_metrics = []

    for fold in range(5):
        val_idx   = folds_df.loc[folds_df["fold"] == fold, "index"].values
        train_idx = train_val_df.index.difference(val_idx)

        df_tr = train_val_df.loc[train_idx]
        df_va = train_val_df.loc[val_idx]

        ds_tr = Dataset.from_pandas(df_tr).map(tokenize, batched=True)
        ds_va = Dataset.from_pandas(df_va).map(tokenize, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

        args = TrainingArguments(
            output_dir=str(OPTUNA_MODELS_DIR / f"fold_{fold}"),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            learning_rate=lr,
            weight_decay=0.01,
            logging_steps=10,
            report_to="none",
            disable_tqdm=True,
            save_strategy="epoch",
            save_total_limit=1,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_tr,
            eval_dataset=ds_va,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        fold_metrics.append(metrics)

        # Save globally best model by F1 (across all folds & trials)
        if metrics["eval_f1"] > best_f1:
            best_f1 = metrics["eval_f1"]
            best_metrics = {**metrics, "learning_rate": lr, "batch_size": batch_size}
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)

        print(f"Fold {fold} | F1: {metrics['eval_f1']:.4f} | LR: {lr:.1e} | BS: {batch_size}")

    avg_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0].keys()}
    avg_metrics.update({"learning_rate": lr, "batch_size": batch_size})

    with open(RESULTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["trial", "learning_rate", "batch_size", "eval_f1"])
        writer.writerow([trial.number, lr, batch_size, avg_metrics["eval_f1"]])

    return float(avg_metrics["eval_f1"])
# Returns mean F1 across folds (the Optuna objective) and appends each trial to CSV for tracking.

# ==== Run Study ====
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# ==== Save Best Trial Summary ====
best_trial = study.best_trial
best_trial_row = [{
    "trial":         best_trial.number,
    "learning_rate": best_trial.params["learning_rate"],
    "batch_size":    best_trial.params["batch_size"],
    "f1":            best_trial.value,
}]
pd.DataFrame(best_trial_row).to_csv(BEST_TRIAL_FILE, index=False)

print("\nBest trial:", best_trial.number)
print("Best value (F1):", best_trial.value)
print("Best params:", best_trial.params)
print(f"Saved best model to: {BEST_MODEL_DIR}")
print(f"Trials log:          {RESULTS_FILE}")
print(f"Best trial summary:  {BEST_TRIAL_FILE}")
