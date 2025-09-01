from pathlib import Path
from dotenv import load_dotenv
import os, praw, json, time

# --- Project setup ---
ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from .env file (API credentials for Reddit)
load_dotenv(ROOT / ".env", override=True)

# Directory where raw data will be stored
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Target subreddit to scrape
SUBREDDIT = "suicidewatch"

# Initialize Reddit API client using PRAW
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT"),
)

# Input (existing posts) and output (new posts) files
EXISTING_FILE = RAW_DIR / "suicidewatch_posts.json"
NEW_FILE = RAW_DIR / "suicidewatch_new_posts.json"

# --- Load existing IDs if file exists ---
existing_ids = set()

if EXISTING_FILE.exists():
    with EXISTING_FILE.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        # Existing file contains a list of posts
        existing_ids = {p["id"] for p in obj if isinstance(p, dict) and "id" in p}
    elif isinstance(obj, dict):
        # Existing file is a dict (id → post object)
        existing_ids = set(obj.keys())
    print(f"Loaded {len(existing_ids)} existing posts.")
else:
    print("No existing file found.")


# --- Function to fetch new posts ---
def fetch_new_posts(subreddit, category, limit, seen_ids):
    """
    Fetch posts from a subreddit for a given category ("new", "hot", "top").
    Skip posts that are already in seen_ids.
    Returns a list of new post dictionaries.
    """
    submissions = getattr(reddit.subreddit(subreddit), category)(limit=limit)
    new_posts = []
    for submission in submissions:
        if submission.id in seen_ids:  # Skip duplicates
            continue
        new_posts.append({
            "id": submission.id,                                   # Unique ID
            "title": submission.title,                             # Title
            "selftext": submission.selftext,                       # Post body
            "score": submission.score,                             # Score
            "upvotes": submission.ups,                             # Number of upvotes
            "num_comments": submission.num_comments,               # Comment count
            "created_datetime": time.strftime('%Y-%m-%d %H:%M:%S',
                                              time.gmtime(submission.created_utc)),  # UTC timestamp
            "subreddit": subreddit,                                # Subreddit name
            "category": category                                   # Category ("new", "hot", "top")
        })
        seen_ids.add(submission.id)  # Mark as seen
    return new_posts


# --- Parameters for collection ---
TARGET_COUNT = 1000   # Total number of posts to collect
TOTAL_NEW = 0         # Counter of collected posts
BATCH_SIZE = 100      # Number of posts to fetch per category per round
MAX_ROUNDS = 15       # Safety cap on rounds

all_new_posts = []
rounds = 0

# --- Main collection loop ---
while TOTAL_NEW < TARGET_COUNT and rounds < MAX_ROUNDS:
    print(f"\nRound {rounds + 1}: Fetching...")
    new_round_posts = []

    for cat in ["new", "hot", "top"]:
        posts = fetch_new_posts(SUBREDDIT, cat, BATCH_SIZE, existing_ids)
        print(f"  {len(posts)} new posts from '{cat}'")
        new_round_posts.extend(posts)

    if not new_round_posts:
        print("No more new posts found.")
        break

    all_new_posts.extend(new_round_posts)
    TOTAL_NEW += len(new_round_posts)
    rounds += 1
    time.sleep(2)  # Sleep to avoid hitting Reddit API rate limits

# --- Save results ---
print(f"\nTotal new posts collected: {len(all_new_posts)}")

with NEW_FILE.open("w", encoding="utf-8") as f:
    json.dump(all_new_posts, f, ensure_ascii=False, indent=2)

print(f"Saved ONLY new posts to '{NEW_FILE}'")
