from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 1) Project paths =====
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Location of final model results
FINAL_ROOT   = PROJECT_ROOT / "src" / "modeling" / "final"

# Output folder for charts
OUT_DIR      = PROJECT_ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 2) Input metric files =====
# Each CSV must contain at least: accuracy, precision, recall, f1
FILES = {
    "SVM (TF-IDF)" : FINAL_ROOT / "SVM"        / "svm_final_metrics.csv",
    "XLNet"        : FINAL_ROOT / "XLNet"      / "xlnet_final_metrics.csv",
    "BERT-base"    : FINAL_ROOT / "BERT_Base"  / "bert_final_metrics.csv",
    "MentalBERT"   : FINAL_ROOT / "MentalBERT" / "mentalbert_final_metrics.csv",
    "RoBERTa"      : FINAL_ROOT / "RoBERTa"    / "roberta_final_metrics.csv",
    "DistilBERT"   : FINAL_ROOT / "DistilBERT" / "distilbert_final_metrics.csv",
}

# ===== 3) Read & aggregate metrics =====
rows, missing = [], []
for name, path in FILES.items():
    if not path.exists():
        # Track missing files but continue
        missing.append((name, str(path)))
        continue
    df = pd.read_csv(path)
    rec = df.iloc[0].to_dict()

    # Ensure required columns exist
    for col in ["precision", "recall", "f1"]:
        if col not in rec:
            raise ValueError(f"Missing required column '{col}' in file: {path}")

    # Collect model metrics
    rows.append({
        "model": name,
        "precision": float(rec["precision"]),
        "recall": float(rec["recall"]),
        "f1": float(rec["f1"]),
        "accuracy": float(rec.get("accuracy", np.nan)),
        "runtime_seconds": float(rec.get("runtime_seconds", np.nan)),
    })

# Merge into DataFrame, sort by F1 descending
metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

# ===== 4) Bar chart: F1 / Recall / Precision =====
x = np.arange(len(metrics_df))
width = 0.25
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=140)

# Grouped bar chart
bars_f1 = ax.bar(x - width, metrics_df["f1"].values,        width, label="F1")
bars_re = ax.bar(x,         metrics_df["recall"].values,    width, label="Recall")
bars_pr = ax.bar(x + width, metrics_df["precision"].values, width, label="Precision")

# Axis formatting
ax.set_xticks(x)
ax.set_xticklabels(metrics_df["model"].tolist(), rotation=0)
ax.set_ylim(0, 1.0)
ax.set_title("Model Comparison on Test — F1 / Recall / Precision")
ax.set_ylabel("Score")
ax.legend(loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.3)

# Annotate bars with values (3 decimals)
def autolabel(bars):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

autolabel(bars_f1)
autolabel(bars_re)
autolabel(bars_pr)

fig.tight_layout()

# Save figure
chart_png = OUT_DIR / "test_models_bar_F1_Recall_Precision.png"
fig.savefig(chart_png, bbox_inches="tight")
plt.close(fig)

print(f" Saved bar chart PNG → {chart_png}")
