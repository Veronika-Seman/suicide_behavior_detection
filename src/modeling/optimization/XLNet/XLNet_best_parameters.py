import pandas as pd
import numpy as np
import torch
import optuna
import csv
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import XLNetTokenizerFast, XLNetForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ===== Project root =====
ROOT = Path(__file__).resolve().parents[4]

# ===== Static input paths =====
SPLITS_DIR     = ROOT / "data" / "processed" / "splits"
TRAINVAL_FIXED = SPLITS_DIR / "train_val_fixed.csv"
FOLDS_FIXED    = SPLITS_DIR / "fold_assignments_fixed.csv"

# ===== Output paths for this model =====
XLNET_DIR          = ROOT / "XLNet"
OPTUNA_MODELS_DIR  = XLNET_DIR / "optuna_models_xlnet"
BEST_MODEL_DIR     = XLNET_DIR / "best_optuna_model_xlnet"
RESULTS_FILE       = XLNET_DIR / "optuna_xlnet_results.csv"
BEST_TRIAL_FILE    = XLNET_DIR / "best_trial_summary_xlnet.csv"

# Create subfolders
for p in [OPTUNA_MODELS_DIR, BEST_MODEL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ===== Model / device =====
model_name = "xlnet-base-cased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ===== Load data =====
assert TRAINVAL_FIXED.exists(), f"Missing: {TRAINVAL_FIXED}"
assert FOLDS_FIXED.exists(),    f"Missing: {FOLDS_FIXED}"

train_val_df = pd.read_csv(TRAINVAL_FIXED)
train_val_df.set_index('index', inplace=True)

folds_df = pd.read_csv(FOLDS_FIXED).astype({'index': int})

# ===== Tokenizer =====
tokenizer = XLNetTokenizerFast.from_pretrained(model_name)

def compute_metrics(pred):
    """
   Computes binary classification metrics:
  - Takes logits and true labels from Hugging Face EvalPrediction.
  - Uses argmax over logits to decide predicted class {0,1}.
  - Returns accuracy, precision, recall, and F1 (with class '1' as positive).
  - "eval_f1" is the main score used to compare trials and pick the best model.
    """
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": p, "recall": r, "eval_f1": f1}

def tokenize(batch):
    """
    Prepare a batch for XLNet: pad to batch max length, truncate to 256 tokens,
    and attach 'labels' so Trainer computes supervised loss automatically.
    """
    tokens = tokenizer(batch['Text'], padding=True, truncation=True, max_length=256)
    tokens["labels"] = batch["Lable"]
    return tokens
# Concise: standard HF batching → (input_ids, attention_mask) + labels for training.

# ===== Global trackers (best across all folds/trials) =====
best_f1 = 0.0
best_metrics = {}

def objective(trial):
    """
    Optuna objective:
      • Samples learning_rate (log-scale) and batch_size.
      • Runs 5-fold CV using the fixed fold indices.
      • Trains XLNet for 3 epochs per fold and evaluates on the fold’s val set.
      • Saves model/tokenizer when a new global best F1 appears.
      • Logs the trial’s mean F1 across folds to CSV and returns it.
    """
    global best_f1, best_metrics

    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    fold_metrics = []

    for fold in range(5):
        val_idx   = folds_df[folds_df['fold'] == fold]['index'].values
        train_idx = train_val_df.index.difference(val_idx)

        df_tr = train_val_df.loc[train_idx]
        df_va = train_val_df.loc[val_idx]

        ds_tr = Dataset.from_pandas(df_tr).map(tokenize, batched=True)
        ds_va = Dataset.from_pandas(df_va).map(tokenize, batched=True)

        model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to(device)

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
            save_total_limit=1
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds_tr,
            eval_dataset=ds_va,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        metrics = trainer.evaluate()
        fold_metrics.append(metrics)

        # Save global best by F1
        if metrics['eval_f1'] > best_f1:
            best_f1 = metrics['eval_f1']
            best_metrics = metrics.copy()
            best_metrics.update({"learning_rate": lr, "batch_size": batch_size})
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)

        print(f"Fold {fold} | F1: {metrics['eval_f1']:.4f} | LR: {lr:.1e} | BS: {batch_size}")

    # Mean metrics across folds for this trial
    avg_metrics = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in fold_metrics[0].keys()}
    avg_metrics.update({"learning_rate": lr, "batch_size": batch_size})

    # Append trial log (create header if file is empty)
    with open(RESULTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["trial", "learning_rate", "batch_size", "eval_f1"])
        writer.writerow([trial.number, lr, batch_size, avg_metrics['eval_f1']])

    return avg_metrics['eval_f1']
# Concise: 5-fold CV; returns mean F1 (objective); persists the best model across all folds/trials.

# ===== Run Optuna =====
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# ===== Save Best Trial Summary =====
best_trial = study.best_trial
best_trial_metrics = {
    "trial": best_trial.number,
    "learning_rate": best_trial.params["learning_rate"],
    "batch_size": best_trial.params["batch_size"],
    "f1": best_trial.value
}
pd.DataFrame([best_trial_metrics]).to_csv(BEST_TRIAL_FILE, index=False)

print("\nBest trial:", best_trial.number)
print("Best value (F1):", best_trial.value)
print("Best params:", best_trial.params)
print(f"Best model dir -> {BEST_MODEL_DIR}")
print(f"Results CSV    -> {RESULTS_FILE}")
