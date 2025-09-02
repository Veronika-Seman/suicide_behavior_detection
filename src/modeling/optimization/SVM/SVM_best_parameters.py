import pandas as pd
import numpy as np
import optuna
import os
import csv
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ===== ROOT & Paths =====
ROOT = Path(__file__).resolve().parents[4]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
TRAINVAL_FIXED = SPLITS_DIR / "train_val_fixed.csv"
FOLDS_FIXED = SPLITS_DIR / "fold_assignments_fixed.csv"

SVM_DIR = ROOT / "SVM"
SVM_DIR.mkdir(parents=True, exist_ok=True)
(SVM_DIR / "optuna_models_svm").mkdir(parents=True, exist_ok=True)

results_file = SVM_DIR / "optuna_svm_results.csv"
best_trial_summary_file = SVM_DIR / "best_trial_summary_svm.csv"

# ===== Load data & folds =====
train_val_df = pd.read_csv(TRAINVAL_FIXED)
folds_df = pd.read_csv(FOLDS_FIXED)

# Ensure correct indexing
train_val_df.set_index('index', inplace=True)
folds_df = folds_df.astype({'index': int})

# ===== Metrics function =====
def compute_metrics(y_true, y_pred):
    """
    Computes binary classification metrics:
  - Takes logits and true labels from Hugging Face EvalPrediction.
  - Uses argmax over logits to decide predicted class {0,1}.
  - Returns accuracy, precision, recall, and F1 (with class '1' as positive).
  - "eval_f1" is the main score used to compare trials and pick the best model.
    """
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# ===== Optuna objective =====
def objective(trial):
    """
    Objective for Optuna hyperparameter optimization of SVM with TF-IDF features.

    Workflow:
      • Sample hyperparameters: C, kernel, gamma (if RBF).
      • Perform 5-fold cross-validation using fixed fold indices.
      • Train TF-IDF + SVM pipeline per fold and evaluate on validation split.
      • Compute metrics and return mean F1 across folds (objective value).
      • Log each trial’s parameters and F1 to CSV.
    """
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
    gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True) if kernel == "rbf" else "scale"

    fold_metrics = []

    for fold in range(5):
        val_idx = folds_df[folds_df["fold"] == fold]["index"].values
        train_idx = train_val_df.index.difference(val_idx)

        train_df = train_val_df.loc[train_idx]
        val_df = train_val_df.loc[val_idx]

        # Build TF-IDF + SVM pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('svm', SVC(C=C, kernel=kernel, gamma=gamma, probability=True))
        ])

        pipeline.fit(train_df['Text'], train_df['Lable'])
        preds = pipeline.predict(val_df['Text'])

        metrics = compute_metrics(val_df['Lable'], preds)
        fold_metrics.append(metrics)

    # Average F1 across folds
    avg_f1 = np.mean([m["f1"] for m in fold_metrics])

    # Append results to CSV
    with open(results_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # header only once
            writer.writerow(["trial", "C", "kernel", "gamma", "f1"])
        writer.writerow([trial.number, C, kernel, gamma if gamma != "scale" else "-", avg_f1])

    return avg_f1
# Explanation: Optuna maximizes the mean F1 across folds; results are logged for later inspection.

# ===== Run Optuna study =====
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# ===== Save best trial =====
best_trial = study.best_trial
best_trial_metrics = {
    "trial": best_trial.number,
    "C": best_trial.params["C"],
    "kernel": best_trial.params["kernel"],
    "gamma": best_trial.params.get("gamma", "-"),
    "f1": best_trial.value
}
pd.DataFrame([best_trial_metrics]).to_csv(best_trial_summary_file, index=False)

print("\nBest trial:", best_trial.number)
print("Best value (F1):", best_trial.value)
print("Best params:", best_trial.params)
print(f"Results CSV  -> {results_file}")
print(f"Summary CSV  -> {best_trial_summary_file}")
