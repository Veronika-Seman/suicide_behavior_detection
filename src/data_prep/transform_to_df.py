from pathlib import Path
import json
import pandas as pd

# Define project root
ROOT = Path(__file__).resolve().parents[2]

# Define raw and interim directories
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

# Input JSON file and output CSV file
IN_JSON  = RAW / "subreddits_combined2.json"
OUT_CSV  = INTERIM / "data1.csv"

# --- Load JSON data ---
with IN_JSON.open("r", encoding="utf-8") as f:
    data = json.load(f)

# Support both JSON structures:
# - list[dict] → already a list of posts
# - dict[id → post] → convert dictionary values into a list
if isinstance(data, dict):
    data = list(data.values())

# --- Convert to DataFrame and save as CSV ---
df = pd.DataFrame(data)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")  # utf-8-sig ensures Excel compatibility

# --- Confirmation ---
print(f"Saved: {OUT_CSV}")
