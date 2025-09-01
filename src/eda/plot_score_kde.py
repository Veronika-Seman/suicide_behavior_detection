from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]

# Processed data directory
PROCESSED = ROOT / "data" / "processed"

# Figures output directory
FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# --- Load dataset ---
df = pd.read_csv(PROCESSED / "final_combined_dataset.csv")

# Ensure "score" column is numeric, and drop missing values
s = pd.to_numeric(df["score"], errors="coerce").dropna()

# --- KDE Plot (distribution of score) ---
plt.figure(figsize=(10, 5))

# Kernel Density Estimate plot
# (If using an older version of seaborn, replace fill=True with shade=True)
sns.kdeplot(s, fill=True, bw_adjust=1.0)

# Add labels and title
plt.title("Score Distribution (KDE Plot)")
plt.xlabel("Score")
plt.ylabel("Density")
plt.tight_layout()

# --- Save and show ---
out_path = FIGS / "score_kde.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved: {out_path}")
