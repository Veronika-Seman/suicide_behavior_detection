from pathlib import Path
import pandas as pd

# Define project root
ROOT = Path(__file__).resolve().parents[2]

# Define raw, interim, and processed data directories
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

# Ensure interim and processed directories exist
INTERIM.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)


# --- Example 1: Process original dataset (raw) ---
# Load raw data
df_original = pd.read_csv(RAW / "data1.csv")

# Convert "created_datetime" column to datetime
df_original["created_datetime"] = pd.to_datetime(df_original["created_datetime"])

# Extract hour of day from timestamp
df_original["hour"] = df_original["created_datetime"].dt.hour

# Extract weekday name from timestamp
df_original["weekday"] = df_original["created_datetime"].dt.day_name()

# Drop unnecessary columns
df_cleaned = df_original.drop(columns=["created_datetime", "id", "upvotes", "category"])

# Save cleaned selection to interim folder
df_cleaned.to_csv(INTERIM / "selected_data1.csv", index=False)


# --- Example 2: Process generated posts dataset (interim) ---
# Load generated dataset (with metadata already added)
df_original = pd.read_csv(INTERIM / "generated_800_full.csv")

# Drop datetime column (keep only useful fields)
df_cleaned = df_original.drop(columns=["created_datetime"])

# Save cleaned selection to interim folder
df_cleaned.to_csv(INTERIM / "generated_data_selection.csv", index=False)
