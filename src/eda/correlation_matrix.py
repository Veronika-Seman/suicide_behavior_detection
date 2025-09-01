from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]

# Processed data directory
PROCESSED = ROOT / "data" / "processed"

# Figures output directory
FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Input CSV
CSV_IN = PROCESSED / "final_combined_dataset.csv"

# --- Target columns (features of interest) ---
TARGETS = [
    "num_comments", "score",
    "hour_sin", "hour_cos",
    "weekday_Friday", "weekday_Monday", "weekday_Saturday",
    "weekday_Sunday", "weekday_Thursday", "weekday_Tuesday", "weekday_Wednesday",
]

# --- Load and clean column names ---
df = pd.read_csv(CSV_IN)

# Strip extra spaces from column names (if any)
df.columns = df.columns.str.strip()

# Build case-insensitive mapping: lowercase target → actual column name in DataFrame
lower_map = {c.lower(): c for c in df.columns}
selected_cols, missing = [], []
for t in TARGETS:
    c = lower_map.get(t.lower())
    if c:
        selected_cols.append(c)
    else:
        missing.append(t)

# Warn if some target columns are missing in the dataset
if missing:
    print("Warning: Missing columns in dataset:", missing)

# Subset the DataFrame with selected features
X = df[selected_cols].copy()

# Ensure all selected columns are numeric (convert if necessary)
for c in selected_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# --- Diagnostics ---
# Print number of non-null values per column
print("\nNon-null counts:\n", X.notna().sum())

# Detect columns with zero variance (cannot produce meaningful correlations)
zero_var_cols = X.columns[X.std(numeric_only=True) == 0]
if len(zero_var_cols):
    print("Columns with zero variance (will result in NaN correlations):", list(zero_var_cols))

# Drop rows where *all* target columns are NaN (extra safety)
X = X.dropna(how="all", subset=selected_cols)


# --- Correlation matrix and heatmap ---
corr = X.corr(numeric_only=True)

plt.figure(figsize=(12, 9))
sns.heatmap(
    corr, annot=True, fmt=".2f",      # Show correlation values with 2 decimals
    cmap="coolwarm", vmin=-1, vmax=1  # Diverging colormap from -1 to 1
)
plt.title("Correlation Matrix – Selected Features")
plt.tight_layout()

# Save figure as PNG
plt.savefig(FIGS / "correlation_matrix2.png", dpi=220, bbox_inches="tight")

# Display plot
plt.show()
