from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]

# Output directory for figures
FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Input dataset (update path if located elsewhere)
DF_PATH = ROOT / "data" / "interim" / "data1.csv"


# --- Load and preprocess data ---
df = pd.read_csv(DF_PATH)

# Convert created_datetime column to datetime
df["created_datetime"] = pd.to_datetime(df["created_datetime"])

# Extract the hour of day as a separate feature
df["hour"] = df["created_datetime"].dt.hour

# Compute correlation matrix using only numeric columns
corr = df.select_dtypes(include="number").corr()


# --- Plot and save correlation heatmap ---
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,          # Show correlation values inside the heatmap
    cmap="coolwarm",     # Diverging color scale
    fmt=".6f"            # Format numbers with 6 decimal places
)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()

# Save the figure as PNG
plt.savefig(FIGS / "correlation_matrix1.png", dpi=200, bbox_inches="tight")

# Close plot to free memory (important in scripts)
plt.close()
