from pathlib import Path
import pandas as pd

# Define project root and interim data directory
ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

# Input (cleaned dataset) and output (examples for GPT) file paths
CLEANED_DATA = INTERIM / "cleaned_data1.csv"
OUTPUT = INTERIM / "examples_for_gpt.csv"

# --- Load dataset ---
df = pd.read_csv(CLEANED_DATA)

# --- Define keywords to identify suicidal intent ---
keywords = [
    "suicide", "kill myself", "want to die", "end it all", "i'm done",
    "can't go on", "take my life", "no way out", "give up", "ending it"
]

# --- Filter posts ---
filtered = df[
    (df["subreddit"].str.lower() == "suicidewatch") &   # Only from SuicideWatch
    df["selftext"].notna() &                            # Must have selftext
    df["title"].notna() &                               # Must have a title
    (
        df["title"].str.lower().str.contains('|'.join(keywords)) |   # Keywords in title
        df["selftext"].str.lower().str.contains('|'.join(keywords))  # Or in selftext
    )
]

# --- Select examples ---
# Keep only title + selftext columns, drop duplicates, and sample 80 posts reproducibly
samples = filtered[["title", "selftext"]].drop_duplicates().sample(80, random_state=42)

# --- Save results ---
samples.to_csv(OUTPUT, index=False, encoding="utf-8")

print(f"Saved 80 posts from SuicideWatch to '{OUTPUT}'")
