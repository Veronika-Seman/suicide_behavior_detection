from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FINAL_ROOT   = PROJECT_ROOT / "src" / "modeling" / "final"

# Output directory for figures
OUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Choose model and its predictions file ---
MODEL_NAME = "DistilBERT"
PRED_CSV   = FINAL_ROOT / MODEL_NAME / "distilbert_test_predictions.csv"

# --- Helper: load true labels & predictions with flexible column detection ---
def load_y_true_pred(csv_path: Path):
    df = pd.read_csv(csv_path)

    # Auto-detect true label column
    label_cols = [c for c in df.columns if str(c).lower() in {"lable", "label", "y_true", "target"}]
    pred_cols  = [c for c in df.columns if str(c).lower() in {"prediction", "y_pred", "pred"}]

    if not label_cols:
        raise ValueError(f"Missing true-label column in: {csv_path}")
    if not pred_cols:
        raise ValueError(f"Missing prediction column in: {csv_path}")

    y_true = df[label_cols[0]].astype(int).to_numpy()
    y_pred = df[pred_cols[0]].astype(int).to_numpy()
    return y_true, y_pred

# Load ground truth and predictions
y_true, y_pred = load_y_true_pred(PRED_CSV)

# --- Compute recall components (restricted to actual positives) ---
pos_mask = (y_true == 1)   # Only consider true positives
n_pos = int(pos_mask.sum())
tp = int(np.logical_and(pos_mask, y_pred == 1).sum())  # Correctly predicted positives
fn = int(np.logical_and(pos_mask, y_pred == 0).sum())  # Missed positives

# --- Build pie chart (TP vs FN among actual positives) ---
labels = [f"TP (caught) — {tp}", f"FN (missed) — {fn}"]
sizes  = [tp, fn]
colors = ["#1f77b4", "#ff7f0e"]  # Blue for TP, Orange for FN
explode = (0.02, 0.02)           # Slight separation between slices

wedges, texts, autotexts = plt.pie(
    sizes,
    labels=None,                 # Labels handled in legend instead
    colors=colors,
    startangle=90,               # Rotate so first wedge starts at top
    counterclock=False,
    explode=explode,
    autopct=lambda pct: f"{pct:.1f}%",   # Show percentage on wedges
    pctdistance=0.75,
    textprops=dict(color="white", fontsize=12, fontweight="bold")
)

# Style: make wedge borders visible
for w in wedges:
    w.set_linewidth(1.5)
    w.set_edgecolor("white")

# Title and legend (legend positioned on the right)
plt.title(f"{MODEL_NAME} — Recall on Test (actual positives n+={n_pos})", pad=12, fontsize=13)
plt.legend(wedges, labels, title="Segments", loc="center left", bbox_to_anchor=(1, 0.5))

# --- Save figure ---
out_png = OUT_DIR / f"{MODEL_NAME.lower()}_recall_pie.png"
plt.savefig(out_png, bbox_inches="tight", dpi=140)
plt.close()
print(f" Saved: {out_png}")
