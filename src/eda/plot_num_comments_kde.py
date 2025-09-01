from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]

# Processed data folder
PROCESSED = ROOT / "data" / "processed"

# Figures output folder
FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# --- Load dataset ---
df = pd.read_csv(PROCESSED / "final_combined_dataset.csv")

# Ensure "num_comments" column is numeric, and drop NaN values
s = pd.to_numeric(df["num_comments"], errors="coerce").dropna()

# --- KDE Plot (distribution of num_comments) ---
plt.figure(figsize=(10, 5))

# Kernel Density Estimate plot
# (If using older seaborn, replace fill=True with shade=True)
sns.kdeplot(s, fill=True, bw_adjust=1.0)

# Add labels and title
plt.title("num_comments Distribution (KDE Plot)")
plt.xlabel("num_comments")
plt.ylabel("Density")
plt.tight_layout()

# --- Save and display ---
out_path = FIGS / "num_comments_kde.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Saved: {out_path}")
