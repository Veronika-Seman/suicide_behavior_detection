from pathlib import Path
import json

# Create the root path of the project (two directories above this file)
ROOT = Path(__file__).resolve().parents[2]

# Path to the raw data folder
RAW = ROOT / "data" / "raw"

# Path to the file containing existing posts
EXISTING_PATH = RAW / "subreddits_combined.json"

# Path to the file containing newly fetched posts
NEW_PATH = RAW / "suicidewatch_new_posts.json"

# Path to the output file – where the filtered unique posts will be saved
OUT_PATH = RAW / "suicidewatch_new_posts_filtered.json"

# Load the existing posts and extract a set of unique IDs
with EXISTING_PATH.open("r", encoding="utf-8") as f:
    existing_posts = json.load(f)
existing_ids = {post["id"] for post in existing_posts}

# Load the newly fetched posts
with NEW_PATH.open("r", encoding="utf-8") as f:
    new_posts = json.load(f)

# Filter the new posts – keep only posts whose IDs are not already in the existing dataset
unique_new_posts = [post for post in new_posts if post.get("id") not in existing_ids]

# Save the unique new posts into the output file, formatted as a readable JSON
with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(unique_new_posts, f, ensure_ascii=False, indent=2)

# Print a message with the number of unique posts saved
print(f" Cleaned new posts. Saved {len(unique_new_posts)} unique posts to '{OUT_PATH}'.")
