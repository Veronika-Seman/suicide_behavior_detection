from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Project root ---
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# --- Input files ---
# Ensemble metrics (long format, contains metrics per class)
ENSEMBLE_CSV = PROJECT_ROOT / "data" / "Ensemble" / "ensemble_metrics.csv"

# Best single model metrics (wide format, single row with precision/recall/f1/accuracy)
BEST_CSV     = PROJECT_ROOT / "src" / "modeling" / "final" / "DistilBERT" / "distilbert_final_metrics.csv"

# Display name for the best model
BEST_NAME = "Best Model (DistilBERT)"

# --- Formatting options ---
DECIMALS = 3                           # Number of decimals in annotations
FMT = "{:." + str(DECIMALS) + "f}"     # Format string for values


# ---------- Helper functions ----------
def get_row_val_long(df: pd.DataFrame, metric_name: str) -> float:
    """Extract a metric value from a long-format metrics CSV by metric name."""
    row = df.loc[df["metric"].str.lower() == metric_name.lower()]
    if row.empty:
        raise KeyError(f"Missing metric '{metric_name}' in long-format metrics file.")
    return float(row.iloc[0]["value"])


def load_ensemble_macro(path: Path) -> dict:
    """
    Load ensemble metrics (long format) and compute macro averages.
    Returns a dict with Precision, Recall, F1, Accuracy.
    """
    df = pd.read_csv(path)
    return {
        "Precision": (get_row_val_long(df, "precision_class0") + get_row_val_long(df, "precision_class1")) / 2,
        "Recall":    (get_row_val_long(df, "recall_class0")    + get_row_val_long(df, "recall_class1"))    / 2,
        "F1":        (get_row_val_long(df, "f1_class0")        + get_row_val_long(df, "f1_class1"))        / 2,
        "Accuracy":   get_row_val_long(df, "accuracy"),
    }


def load_best_wide(path: Path) -> dict:
    """
    Load metrics for the best individual model (wide format, single row).
    Returns a dict with Precision, Recall, F1, Accuracy.
    """
    row = pd.read_csv(path).iloc[0]
    lower = {c.lower(): c for c in row.index}

    def pick(name: str) -> float:
        return float(row[lower.get(name.lower(), name)])

    return {
        "Precision": pick("precision"),
        "Recall":    pick("recall"),
        "F1":        pick("f1"),
        "Accuracy":  pick("accuracy"),
    }


def annotate(ax, bars, vals, fmt: str = FMT) -> None:
    """Annotate bars with numeric values above them."""
    for b, v in zip(bars, vals):
        ax.annotate(fmt.format(float(v)),
                    (b.get_x() + b.get_width() / 2, float(v)),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", va="bottom", fontsize=10, weight="bold")


# ---------- Main execution ----------
def main():
    # Validate existence of input files
    assert ENSEMBLE_CSV.exists(), f"Ensemble metrics file not found:\n{ENSEMBLE_CSV}"
    assert BEST_CSV.exists(),     f"Best model metrics file not found:\n{BEST_CSV}"

    # Load metrics
    ens  = load_ensemble_macro(ENSEMBLE_CSV)
    best = load_best_wide(BEST_CSV)

    # Metrics order
    metrics = ["F1", "Recall", "Precision", "Accuracy"]
    ens_vals  = np.array([ens[m]  for m in metrics], dtype=float)
    best_vals = np.array([best[m] for m in metrics], dtype=float)

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=140)

    x = np.arange(len(metrics))
    width = 0.36

    # Bar plots
    bars_best = ax.bar(x - width / 2, best_vals, width, label=BEST_NAME, color="#1f77b4")
    bars_ens  = ax.bar(x + width / 2, ens_vals,  width, label="Ensemble (macro avg.)", color="#ff7f0e")

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Score")
    ax.set_title("Ensemble (Macro Avg) vs. Best Model — Metrics Comparison", pad=8)

    # Legend under the title
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 0.99), handletextpad=0.8,
              columnspacing=1.6, borderaxespad=0.0)

    # Light grid
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.5)

    # Annotate bars with values
    annotate(ax, bars_best, best_vals)
    annotate(ax, bars_ens,  ens_vals)

    fig.tight_layout()

    # --- Save to reports/figures ---
    out_dir = PROJECT_ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "ensemble_vs_best_model.png"

    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved: {out}")


if __name__ == "__main__":
    main()
