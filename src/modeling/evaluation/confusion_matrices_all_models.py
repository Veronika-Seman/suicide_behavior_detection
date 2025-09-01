from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ===== 1) Project-relative paths =====
# This file should live under: src/modeling/<evaluation|ensemble>/
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_ROOT   = PROJECT_ROOT / "src" / "modeling" / "final"

# Prediction CSVs located under: src/modeling/final/<MODEL>/<file>.csv
PRED_FILES = {
    "DistilBERT":   FINAL_ROOT / "DistilBERT" / "distilbert_test_predictions.csv",
    "RoBERTa":      FINAL_ROOT / "RoBERTa"    / "roberta_test_predictions.csv",
    "MentalBERT":   FINAL_ROOT / "MentalBERT" / "mentalbert_test_predictions.csv",
    "BERT-base":    FINAL_ROOT / "BERT_Base"  / "bert_test_predictions.csv",
    "XLNet":        FINAL_ROOT / "XLNet"      / "xlnet_test_predictions.csv",
    "SVM (TF-IDF)": FINAL_ROOT / "SVM"        / "svm_test_predictions.csv",
}

# ===== 2) Output directory =====
OUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 3) Fixed colors per confusion matrix cell =====
COLORS = {
    "TN": "#1f77b4",  # blue
    "TP": "#2ca02c",  # green
    "ERR": "#ff7f0e", # orange (used for both FP and FN)
}
TEXT_COLOR = "white"  # text color over solid fills

# ===== 4) Helper functions =====
def load_y_true_pred(csv_path: Path):
    """
    Load true labels and predictions from a CSV file.
    Supports flexible column names:
      - Labels: 'Lable', 'Label', 'y_true', 'target'
      - Predictions: 'prediction', 'y_pred', 'pred'
    """
    df = pd.read_csv(csv_path)

    # Auto-detect true-label column
    label_cols = [c for c in df.columns if str(c).lower() in {"lable", "label", "y_true", "target"}]
    pred_cols  = [c for c in df.columns if str(c).lower() in {"prediction", "y_pred", "pred"}]

    if not label_cols:
        raise ValueError(f"Missing true-label column in: {csv_path} (expected one of: Lable/Label/y_true/target)")
    if not pred_cols:
        raise ValueError(f"Missing prediction column in: {csv_path} (expected one of: prediction/y_pred/pred)")

    y_true = df[label_cols[0]].astype(int).to_numpy()
    y_pred = df[pred_cols[0]].astype(int).to_numpy()
    return y_true, y_pred


def compute_cm(y_true, y_pred):
    """
    Compute a 2x2 confusion matrix:
    [[TN, FP],
     [FN, TP]]
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int)


def plot_confusion_matrix_colors(cm, model_name, out_path, dpi=140):
    """
    Plot a confusion matrix with fixed colors per cell type:
    - TN: blue, TP: green, FP/FN: orange
    Only raw counts (no percentages) are displayed.
    """
    fig, ax = plt.subplots(figsize=(5.0, 4.6), dpi=dpi)

    # Axis labels and ticks
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"]); ax.set_yticklabels(["0", "1"])
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"{model_name} — Confusion Matrix (Test)", pad=12)

    # Fixed color layout
    cell_colors = np.array([
        [COLORS["TN"],  COLORS["ERR"]],  # [TN, FP]
        [COLORS["ERR"], COLORS["TP"]],   # [FN, TP]
    ])

    # Draw colored cells with values
    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                 facecolor=cell_colors[i, j],
                                 edgecolor="white", linewidth=2)
            ax.add_patch(rect)
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color=TEXT_COLOR)

    # Clean layout
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ===== 5) Generate confusion matrices for all models =====
missing = []
for model, path in PRED_FILES.items():
    if not path.exists():
        missing.append((model, str(path)))
        continue

    y_true, y_pred = load_y_true_pred(path)
    cm = compute_cm(y_true, y_pred)

    # Safe filename: lowercase, no spaces, no parentheses/dashes
    safe = (model.lower()
                  .replace(" ", "_")
                  .replace("(", "")
                  .replace(")", "")
                  .replace("-", "_"))
    out_png = OUT_DIR / f"cm_{safe}.png"

    plot_confusion_matrix_colors(cm, model, out_png)
    print(f" Saved: {out_png}")
