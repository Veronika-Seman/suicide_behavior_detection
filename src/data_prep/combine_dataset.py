import pandas as pd
from pathlib import Path

# --- Project setup ---
# Define project root (two levels up from this file)
ROOT = Path(__file__).resolve().parents[2]

# Define the "processed" data directory
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn’t exist

# Input CSV files
DATA1 = PROCESSED / "final_prepared_data1.csv"     # Real prepared dataset
DATA2 = PROCESSED / "final_generated_data.csv"     # Generated (synthetic) dataset

# Output CSV file (merged dataset)
COMBINED_CSV = PROCESSED / "final_combined_dataset.csv"

# --- Load data ---
real_df = pd.read_csv(DATA1)        # Load real dataset into a DataFrame
generated_df = pd.read_csv(DATA2)   # Load generated dataset into a DataFrame

# --- Combine datasets ---
# Concatenate both DataFrames into one, resetting the index
combined_df = pd.concat([real_df, generated_df], ignore_index=True)

# --- Save the result ---
combined_df.to_csv(COMBINED_CSV, index=False, encoding="utf-8")

# Print confirmation message
print(f"Combined dataset saved as '{COMBINED_CSV}'")
