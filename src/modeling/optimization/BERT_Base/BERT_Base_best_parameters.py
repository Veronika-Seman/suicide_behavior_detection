import pandas as pd
import numpy as np
import torch
import optuna
import csv
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ===== ROOT & PATHS =====
ROOT = Path(__file__).resolve().parents[4]

# ===== Paths =====
SPLITS_DIR   = ROOT / "data" / "processed" / "splits"
TRAINVAL_CSV = SPLITS_DIR / "train_val.csv"
FOLDS_CSV    = SPLITS_DIR / "fold_assignments.csv"

# Output directories/files for this experiment
BERT_BASE_DIR     = ROOT / "BERT_Base"
OPTUNA_MODELS_DIR = BERT_BASE_DIR / "optuna_models"
BEST_MODEL_DIR    = BERT_BASE_DIR / "best_optuna_model"
RESULTS_FILE      = BERT_BASE_DIR / "optuna_trial_results.csv"
BEST_TRIAL_FILE   = BERT_BASE_DIR / "best_trial_summary.csv"

# Create subfolders
for p in [OPTUNA_MODELS_DIR, BEST_MODEL_DIR, BERT_BASE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# ===== Model & Device =====
model_name = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# ===== Load prepared splits =====
assert TRAINVAL_CSV.exists(), f"Missing: {TRAINVAL_CSV}"
assert FOLDS_CSV.exists(),    f"Missing: {FOLDS_CSV}"

train_val_df = pd.read_csv(TRAINVAL_CSV)
folds_df = pd.read_csv(FOLDS_CSV).astype({"index": int})

# ===== Tokenizer =====
tokenizer = BertTokenizerFast.from_pretrained(model_name)


def compute_metrics(pred):
    """
    Calculate Accuracy/Precision/Recall/F1 for binary classification
    from Hugging Face `EvalPrediction` (logits -> argmax -> {0,1}).
    Returns a dict with keys: accuracy, precision, recall, eval_f1.
    """
    labels = pred.label_ids
    preds  = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"accuracy": acc, "precision": p, "recall": r, "eval_f1": f1}
# Uses argmax over logits; treats class "1" as positive (`average="binary"`); `eval_f1` is the score Optuna maximizes.


def tokenize(batch):
    """
    Prepare a batch for BERT: pad/truncate texts to 256 tokens and
    attach labels so the Trainer can compute supervised loss.
    """
    t = tokenizer(batch["Text"], padding=True, truncation=True, max_length=256)
    t["labels"] = batch["Lable"]
    return t
# Padding keeps batch tensors aligned; truncation caps overly long posts; adding "labels" enables built-in loss.


# ===== Global trackers =====
best_f1 = 0.0
best_metrics = {}


def objective(trial):
    """
    Optuna objective:
    sample learning_rate & batch_size, run 5-fold CV using precomputed
    indices, save the globally best model (by F1), and return the mean F1.
    Also appends trial results to `optuna_trial_results.csv`.
    """
    global best_f1, best_metrics

    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    fold_metrics = []

    # Keep index alignment stable for fold masking
    tv = train_val_df.copy()
    tv.set_index(tv.index, inplace=True)

    for fold in range(5):
        val_idx   = folds_df.loc[folds_df["fold"] == fold, "index"].values
        train_idx = tv.index.difference(val_idx)

        df_tr = tv.loc[train_idx]
        df_va = tv.loc[val_idx]

        ds_tr = Dataset.from_pandas(df_tr).map(tokenize, batched=True)
        ds_va = Dataset.from_pandas(df_va).map(tokenize, batched=True)

        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

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

        # Track global best across all trials/folds
        if metrics["eval_f1"] > best_f1:
            best_f1 = metrics["eval_f1"]
            best_metrics = {**metrics, "learning_rate": lr, "batch_size": batch_size}
            model.save_pretrained(BEST_MODEL_DIR)
            tokenizer.save_pretrained(BEST_MODEL_DIR)

        print(f"Fold {fold} | F1: {metrics['eval_f1']:.4f} | LR: {lr:.1e} | BS: {batch_size}")

    avg_f1 = float(np.mean([m["eval_f1"] for m in fold_metrics]))

    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["trial", "learning_rate", "batch_size", "eval_f1"])
        w.writerow([trial.number, lr, batch_size, avg_f1])

    return avg_f1
# 5-fold CV over train_val; returns mean F1; saves the best-so-far model/tokenizer; logs each trial to CSV.


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    best_trial = study.best_trial
    pd.DataFrame([{
        "trial": best_trial.number,
        "learning_rate": best_trial.params["learning_rate"],
        "batch_size": best_trial.params["batch_size"],
        "f1": best_trial.value,
    }]).to_csv(BEST_TRIAL_FILE, index=False)

    print("\nBest trial:", best_trial.number)
    print("Best value (F1):", best_trial.value)
    print("Best params:", best_trial.params)
    print(f"Saved best model to → {BEST_MODEL_DIR}")
    print(f"Trials log         → {RESULTS_FILE}")
    print(f"Best trial summary → {BEST_TRIAL_FILE}")


if __name__ == "__main__":
    main()
