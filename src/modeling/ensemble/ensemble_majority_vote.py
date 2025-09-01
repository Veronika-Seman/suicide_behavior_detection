from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# --- Project root ---
# (This script is inside: src/modeling/ensemble/… so go up 3 levels)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# --- Input files ---
# Test set with ground-truth labels
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv"

# Prediction outputs from individual models
MODEL_FILES = [
    PROJECT_ROOT / "src" / "modeling" / "final" / "DistilBERT" / "distilbert_test_predictions.csv",
    PROJECT_ROOT / "src" / "modeling" / "final" / "MentalBERT" / "mentalbert_test_predictions.csv",
    PROJECT_ROOT / "src" / "modeling" / "final" / "RoBERTa"   / "roberta_test_predictions.csv",
]

# --- Output files ---
OUT_DIR     = PROJECT_ROOT / "data" / "ensemble"
OUT_MERGED  = OUT_DIR / "ensemble_by_index.csv"   # merged predictions with ensemble results
OUT_METRICS = OUT_DIR / "ensemble_metrics.csv"    # performance metrics


# --- Column definitions ---
TEST_ID_COL   = "index"     # in test.csv
MODEL_ID_COL  = "post_id"   # in model prediction CSVs
TEXT_COL      = "Text"
YTRUE_COL     = "Lable"     # if test.csv uses "Label", adjust here
PRED_COL      = "prediction"


def require(df: pd.DataFrame, cols: list[str], name: str):
    """Ensure DataFrame contains required columns, else raise an error."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}. Found: {list(df.columns)}")


# --- Check input existence ---
assert TEST_PATH.exists(), f"test.csv not found: {TEST_PATH}"
for p in MODEL_FILES:
    assert p.exists(), f"Prediction file not found: {p}"


# --- Load test set (ground truth) ---
gold = pd.read_csv(TEST_PATH)
require(gold, [TEST_ID_COL, TEXT_COL, YTRUE_COL], "test")

# Normalize test index column
gold = gold.rename(columns={TEST_ID_COL: "index"})
merged = gold[["index", TEXT_COL, YTRUE_COL]].copy()


# --- Load model predictions and merge ---
for i, path in enumerate(MODEL_FILES, start=1):
    df = pd.read_csv(path)
    require(df, [MODEL_ID_COL, PRED_COL], f"model{i}")
    df_i = df[[MODEL_ID_COL, PRED_COL]].rename(columns={
        MODEL_ID_COL: "index",
        PRED_COL:     f"pred_m{i}"
    })
    merged = merged.merge(df_i, on="index", how="inner")


# --- Majority vote ensemble ---
vote_cols = [f"pred_m{i}" for i in range(1, len(MODEL_FILES) + 1)]
merged[vote_cols] = merged[vote_cols].astype(int)
merged[YTRUE_COL] = merged[YTRUE_COL].astype(int)

# Sum of votes across models
votes_sum = merged[vote_cols].sum(axis=1)

# Required majority (e.g. 2/3 for 3 models)
num_models = len(vote_cols)
needed = (num_models // 2) + 1

# Ensemble prediction (1 if majority >= needed, else 0)
merged["y_pred_ensemble"] = (votes_sum >= needed).astype(int)

# Agreement ratio (e.g. "2/3", "3/3")
agree_count = np.where(
    merged["y_pred_ensemble"] == 1,
    votes_sum,
    num_models - votes_sum
).astype(int)
merged["agree_ratio"] = pd.Series(agree_count, index=merged.index).map(lambda x: f"{x}/{num_models}")


# --- Evaluation metrics ---
y_true = merged[YTRUE_COL].to_numpy()
y_hat  = merged["y_pred_ensemble"].to_numpy()

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_hat, labels=[0, 1], zero_division=0
)
f1_macro = float(np.mean(f1))

cm  = confusion_matrix(y_true, y_hat, labels=[0, 1])
acc = accuracy_score(y_true, y_hat)
report = classification_report(y_true, y_hat, digits=4, zero_division=0)

TN, FP, FN, TP = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])


# --- Save outputs ---
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Save merged predictions with ensemble results
keep_cols = ["index", TEXT_COL, YTRUE_COL] + vote_cols + ["y_pred_ensemble", "agree_ratio"]
merged[keep_cols].to_csv(OUT_MERGED, index=False, encoding="utf-8")

# Save metrics to CSV
pd.DataFrame([
    ("accuracy", acc), ("f1_macro", f1_macro),
    ("precision_class0", precision[0]), ("recall_class0", recall[0]), ("f1_class0", f1[0]),
    ("precision_class1", precision[1]), ("recall_class1", recall[1]), ("f1_class1", f1[1]),
    ("support_class0", int(support[0])), ("support_class1", int(support[1])),
    ("total_samples", int(len(y_true))), ("TN", TN), ("FP", FP), ("FN", FN), ("TP", TP),
], columns=["metric", "value"]).to_csv(OUT_METRICS, index=False, encoding="utf-8")


# --- Print summary ---
print("=== Ensemble (Majority Vote) ===")
print(f"Accuracy:   {acc:.4f}")
print(f"F1 Macro:   {f1_macro:.4f}")
print("Confusion Matrix [rows=true, cols=pred]:\n", cm)
print("\nClassification Report:\n", report)
print(f"\nSaved merged predictions to: {OUT_MERGED}")
print(f"Saved metrics to:            {OUT_METRICS}")
