import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# --- Project setup ---
ROOT = Path(__file__).resolve().parents[2]

# Directories for processed and interim data
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)
INTERIM = ROOT / "data" / "interim"

# Load environment variables (API keys, etc.)
load_dotenv(ROOT / ".env", override=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Output files
OUT_CSV = PROCESSED / "final_generated_data.csv"
OUT_PKL = PROCESSED / "final_generated_data.pkl"


# --- Functions ---

def generate_selftext(title):
    """
    Generate a short, emotional Reddit-style selftext for a given post title
    using the OpenAI API (chat completion).
    Returns generated text or None if the request fails.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Reddit user writing personal and emotional self-posts. "
                                              "Continue the thought expressed in the title in a realistic, first-person voice. "
                                              "Write as if this is your own post."},
                {"role": "user", "content": f"Write a short selftext (1–3 emotional sentences) based on this Reddit post title:\n'{title}'"}
            ],
            temperature=0.8,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating for '{title}': {e}")
        return None


def is_invalid_text(text):
    """
    Check if the given selftext is invalid:
    - Empty, too short, or NaN
    - Contains only symbols/emojis/placeholder values
    - Starts with a URL
    """
    if pd.isna(text):
        return True
    text = str(text).strip()
    invalid_texts = ["...", "!!", "??", ".", "😔", "😞", "🤷", "¯\\_(ツ)_/¯", "…", "?", "⠀", "unknown"]
    return len(text) < 10 or text in invalid_texts or text.startswith("http")


def normalize_columns_minmax(df, columns):
    """
    Normalize numerical columns (e.g., score, num_comments) to a 0–1 range using Min-Max scaling.
    """
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def cyclical_encode_hour(df, hour_column="hour"):
    """
    Encode hour-of-day as cyclical features (sine and cosine).
    This prevents treating hours as linear values.
    """
    df["hour_sin"] = np.sin(2 * np.pi * df[hour_column] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df[hour_column] / 24)
    df.drop(columns=[hour_column], inplace=True)
    print("Cyclical encoding applied to 'hour'")
    return df


def encode_weekday_onehot(df):
    """
    Apply one-hot encoding to the 'weekday' column if present.
    Converts weekdays into separate binary columns.
    """
    if "weekday" in df.columns:
        df = pd.get_dummies(df, columns=["weekday"], prefix="weekday")
        print("One-hot encoding applied to 'weekday'")
    else:
        print("'weekday' column not found. Skipping encoding.")
    return df


def tokenize_multilingual(df, max_length=512):
    """
    Tokenize combined text (title + selftext) using multilingual BERT tokenizer.
    Adds 'input_ids' and 'attention_mask' columns to the DataFrame.
    """
    if "full_text" not in df.columns:
        df["full_text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
    tokens = tokenizer(
        list(df["full_text"]),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    df["input_ids"] = tokens["input_ids"].tolist()
    df["attention_mask"] = tokens["attention_mask"].tolist()

    print(f"Tokenization complete for {len(df)} posts using multilingual BERT.")
    return df


# --- Main pipeline ---
def main():
    # Load dataset
    df = pd.read_csv(INTERIM / "cleaned_generated_data.csv")
    print(f" Loaded dataset with {len(df)} rows.")

    # Ensure 'generation' column exists (marks which posts are AI-generated)
    if "generation" not in df.columns:
        df["generation"] = 0

    # Identify posts with missing or invalid selftext
    problematic = df[df["selftext"].apply(is_invalid_text)].copy()
    print(f" Found {len(problematic)} posts with missing or invalid selftext. Starting generation...\n")

    # Fill invalid posts by generating new selftexts with GPT
    generated_count = 0
    for index, row in tqdm(problematic.iterrows(), total=problematic.shape[0],
                           desc="Generating selftext", ncols=100):
        title = row["title"]
        new_text = generate_selftext(title)
        if new_text:
            df.at[index, "selftext"] = new_text
            df.at[index, "generation"] = 1
            generated_count += 1

    print(f" Total posts generated (filled): {generated_count}")

    # Normalize numerical features
    df = normalize_columns_minmax(df, ["score", "num_comments"])

    # Encode temporal features
    df = cyclical_encode_hour(df, "hour")
    df = encode_weekday_onehot(df)

    # Tokenize text for modeling
    df = tokenize_multilingual(df)

    # Save processed dataset in CSV and Pickle formats
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved as '{OUT_CSV}'")

    df.to_pickle(OUT_PKL)
    print(f"Saved as '{OUT_PKL}'")


# --- Entry point ---
if __name__ == "__main__":
    main()
