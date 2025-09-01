import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path

# --- Project setup ---
ROOT = Path(__file__).resolve().parents[2]

# Directory for interim data
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

# Input and output file paths
GENERATED_CSV = INTERIM / "generated_800_raw.csv"   # Raw generated posts
CLEANED_DATA = INTERIM / "cleaned_data1.csv"        # Real cleaned dataset (reference for enrichment)
OUTPUT = INTERIM / "generated_800_full.csv"         # Enriched generated posts

# --- Load datasets ---
df_generated = pd.read_csv(GENERATED_CSV)   # Raw generated posts
df_real = pd.read_csv(CLEANED_DATA)         # Real dataset for sampling distributions

# --- Enrich generated posts with realistic metadata ---

# Randomly sample scores from real posts
df_generated["score"] = random.choices(df_real["score"].dropna().tolist(), k=len(df_generated))

# Randomly sample number of comments from real posts
df_generated["num_comments"] = random.choices(df_real["num_comments"].dropna().tolist(), k=len(df_generated))

# Assign subreddit name (fixed for all)
df_generated["subreddit"] = "SuicideWatch"

# Mark these posts as generated (AI-generated flag)
df_generated["generation"] = 1

# Assign a random posting hour (0–23)
df_generated["hour"] = [random.randint(0, 23) for _ in range(len(df_generated))]

# Assign a random weekday
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df_generated["weekday"] = [random.choice(weekdays) for _ in range(len(df_generated))]

# Assign random creation dates (within the last 365 days)
today = datetime.today()
df_generated["created_datetime"] = [
    (today - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d %H:%M:%S")
    for _ in range(len(df_generated))
]

# --- Save the enriched dataset ---
df_generated.to_csv(OUTPUT, index=False, encoding="utf-8")
print(f"Saved enriched generated posts to '{OUTPUT}'")
