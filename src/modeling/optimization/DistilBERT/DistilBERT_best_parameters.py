import pandas as pd
import numpy as np
import torch
import optuna
import csv
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset


ROOT = Path(__file__).resolve().parents[4]

# ===== Paths =====
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
TRAINVAL_FIXED = SPLITS_DIR / "train_val_fixed.csv"
FOLDS_FIXED = SPLITS_DIR / "fold_assignments_fixed.csv"

# Output directories
DISTIL_DIR = ROOT / "DistilBERT"
OPTUNA_MODELS_DIR = DISTIL_DIR / "optuna_models"
BEST_MODEL_DIR = DISTIL_DIR / "best_optuna_model"
RESULTS_FILE = DISTIL_DIR / "optuna_trial_results.csv"
BEST_TRIAL_FILE = DISTIL_DIR / "best_trial_summary.csv"


for p in [SPLITS_DIR, OPTUNA_MODELS_DIR, BEST_MODEL_DIR, DISTIL_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ===== General settings =====
model_name = 'distilbert-base-uncased'

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ===== Load static split files =====
assert TRAINVAL_FIXED.exists(), f"Missing: {TRAINVAL_FIXED}"
assert FOLDS_FIXED.exists(), f"Missing: {FOLDS_FIXED}"

train_val_df = pd.read_csv(TRAINVAL_FIXED)
if 'index' in train_val_df.columns:
    train_val_df.set_index('index', inplace=True)

folds_df = pd.read_csv(FOLDS_FIXED).astype({'index': int})

# ===== Tokenizer =====
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# ===== Metrics =====
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
     Prepares a batch for DistilBERT training:
   - Applies padding so all sequences in the batch have equal length.
   - Truncates sequences longer than 256 tokens.
   - Returns input_ids and attention_mask, and adds "labels" for supervised loss.
    """
    tokens = tokenizer(batch['Text'], padding=True, truncation=True, max_length=256)
    tokens["labels"] = batch["Lable"]
    return tokens


# ===== Global trackers =====
best_f1 = 0
best_metrics = {}


# ===== Optuna objective =====
def objective(trial):
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

    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    fold_metrics = []

    # 5-fold CV
    for fold in range(5):
        val_idx = folds_df[folds_df['fold'] == fold]['index'].values
        train_idx = train_val_df.index.difference(val_idx)

        train_df = train_val_df.loc[train_idx]
        val_df = train_val_df.loc[val_idx]

        train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
        val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)

        model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
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
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        metrics = trainer.evaluate()
        fold_metrics.append(metrics)

        # Track global best model
        if metrics['eval_f1'] > best_f1:
            best_f1 = metrics['eval_f1']
            best_metrics = metrics.copy()
            best_metrics.update({"learning_rate": lr, "batch_size": batch_size})
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)

        print(f"Fold {fold} | F1: {metrics['eval_f1']:.4f} | LR: {lr:.1e} | BS: {batch_size}")

    # Average metrics across folds
    avg_metrics = {k: np.mean([fm[k] for fm in fold_metrics]) for k in fold_metrics[0].keys()}
    avg_metrics.update({"learning_rate": lr, "batch_size": batch_size})

    # Log trial results
    with open(RESULTS_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["trial", "learning_rate", "batch_size", "eval_f1"])
        writer.writerow([trial.number, lr, batch_size, avg_metrics['eval_f1']])

    return avg_metrics['eval_f1']


# ===== Run Optuna study =====
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# ===== Save best trial summary =====
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
print(f"best model dir   -> {BEST_MODEL_DIR}")
print(f"results csv      -> {RESULTS_FILE}")
print(f"best trial csv   -> {BEST_TRIAL_FILE}")
