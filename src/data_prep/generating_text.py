from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import os
import pandas as pd
from tqdm import tqdm
import random

# Define project root
ROOT = Path(__file__).resolve().parents[2]

# Directory for interim data
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

# Input file with example posts, and output file for generated posts
EXAMPLES_CSV = INTERIM / "examples_for_gpt.csv"
OUT_RAW = INTERIM / "generated_800_raw.csv"

# Load environment variables
load_dotenv(ROOT / ".env", override=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load example posts into a list of dictionaries
examples_df = pd.read_csv(EXAMPLES_CSV)
examples_list = examples_df.to_dict(orient="records")


# --- Helper functions ---
def get_random_examples(n=5):
    """
    Select a random sample of n example posts from the dataset.
    """
    return random.sample(examples_list, n)


def generate_post_from_examples():
    """
    Generate a new suicidal-style Reddit post using GPT,
    based on randomly chosen examples for inspiration.
    Returns a string response (Title + Text).
    """
    examples = get_random_examples()

    # Construct conversation messages for the model
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Reddit user in the SuicideWatch subreddit. "
                "You write emotional, personal, and realistic posts that express suicidal thoughts. "
                "You must imitate real Reddit users. Write content that sounds human, distressed, and emotionally detailed. "
                "Use inspiration from the following examples but do not copy them. "
                "You may refer to specific reasons for distress (e.g., family, loneliness, trauma). "
                "Include expressions such as 'want to die', 'suicide', 'end it all', 'no way out', 'im done' or similar phrases."
            ),
        }
    ]

    # Add a few example posts to guide the style
    for i, ex in enumerate(examples, start=1):
        messages.append({
            "role": "user",
            "content": f"Example {i}:\nTitle: {ex['title']}\nText: {ex['selftext']}"
        })

    # Ask the model to create a new post
    messages.append({
        "role": "user",
        "content": "Now write a new post based on the style and emotion of the examples. Give me:\nTitle:\nText:"
    })

    # Request generation from GPT
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.9,   # Higher temperature for creativity
            max_tokens=300     # Limit to short posts
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error:", e)
        return None


# --- Generate 800 new posts ---
generated_data = []

for _ in tqdm(range(800), desc="Generating posts"):
    output = generate_post_from_examples()
    if output and "title" in output.lower():
        lines = output.split("\n")
        title = ""
        text = ""
        for line in lines:
            if line.lower().startswith("title"):
                title = line.split(":", 1)[-1].strip()
            elif line.lower().startswith("text"):
                text = line.split(":", 1)[-1].strip()
            else:
                text += " " + line.strip()
        if title and text:
            generated_data.append({"title": title, "selftext": text})

# --- Save results ---
generated_df = pd.DataFrame(generated_data)

# Save the DataFrame to a CSV file at the path defined by OUT_RAW
generated_df.to_csv(OUT_RAW, index=False, encoding="utf-8")

print(f"Saved 800 generated posts to '{OUT_RAW}'")
