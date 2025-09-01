from pathlib import Path
from dotenv import load_dotenv
import os
import json
import datetime
import praw

# ---- Define paths and load environment variables ----
ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from .env file (API credentials and secrets)
load_dotenv(dotenv_path=ROOT / ".env")

# Create a Reddit object using PRAW, with credentials stored in the .env file
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT")
)

# Directory to save the collected post files
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn’t exist


# ---- Function to fetch posts from a subreddit ----
def fetch_posts(subreddit_name, categories, limit=5000):
    """
    Fetch posts from a given subreddit, for the specified categories ("new", "hot", "top").
    Returns a list of post dictionaries containing key metadata.
    """
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    for category in categories:
        # Choose the fetching method according to the category
        if category == "hot":
            submissions = subreddit.hot(limit=limit)
        elif category == "new":
            submissions = subreddit.new(limit=limit)
        elif category == "top":
            submissions = subreddit.top(limit=limit)
        else:
            continue  # Skip invalid categories

        # Iterate over all fetched submissions and extract relevant details
        for submission in submissions:
            created_datetime = datetime.datetime.utcfromtimestamp(
                submission.created_utc
            ).strftime('%Y-%m-%d %H:%M:%S')

            posts.append({
                "id": submission.id,                         # Unique ID of the post
                "title": submission.title,                   # Post title
                "selftext": submission.selftext,             # Post body text
                "score": submission.score,                   # Post score (upvotes - downvotes)
                "num_comments": submission.num_comments,     # Number of comments
                "category": getattr(submission, "link_flair_text", None),  # Flair category (if exists)
                "subreddit": submission.subreddit.display_name,  # Subreddit name
                "created_datetime": created_datetime         # Post creation timestamp
            })
    return posts


# ---- Main execution ----
if __name__ == "__main__":
    # List of subreddits to collect data from
    subreddits = [
        "anxiety", "depression", "SuicideWatch", "mentalhealth", "EDAnonymous", "healthanxiety",
        "lonely", "ptsd", "MentalHealthUK", "talktherapy", "offmychest",
        "traumatoolbox", "Anger", "dbtselfhelp", "OpiatesRecovery", "selfharm"
    ]

    # Categories to fetch posts from
    categories = ["new", "hot", "top"]

    # Loop through each subreddit and fetch posts
    for sr in subreddits:
        print(f" Fetching posts from r/{sr} ...")
        posts = fetch_posts(sr, categories)

        # Remove duplicate posts by ID
        unique_posts = list({p["id"]: p for p in posts}.values())

        # Save the collected data into a JSON file
        out_path = RAW_DIR / f"{sr}_posts.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(unique_posts, f, ensure_ascii=False, indent=2)

        print(f" Saved {len(unique_posts)} posts to {out_path}")

    print(" Data collection complete. Check the files in data/raw/")
