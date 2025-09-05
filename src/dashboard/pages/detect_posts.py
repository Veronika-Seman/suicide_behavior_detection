import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==== Page Configuration ====

# Set page title and layout
st.set_page_config(page_title="Detect Posts", layout="centered")


# ==== Custom CSS Styling ====

# Apply light background, left-to-right orientation, and left-aligned text.
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
            direction: ltr;
        }
        .main {
            background-color: #f0f8ff;
        }
        h1, h2, h3, h4, h5, h6, p {
            text-align: left;
            direction: ltr;
        }
        .css-18e3th9 {
            background-color: #f0f8ff;
        }
    </style>
""", unsafe_allow_html=True)


# ==== Load Dataset (Portable with ROOT) ====

# Define project root (go up two levels from this file).
ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = ROOT / "data" / "processed" / "final_balanced_dataset.csv"

# Safety check: stop if file not found
if not DATASET_PATH.is_file():
    st.error(f"Dataset not found at: {DATASET_PATH}")
    st.stop()

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Normalize label column name (support both 'Lable' and 'Label')
if 'Lable' in df.columns and 'Label' not in df.columns:
    df = df.rename(columns={'Lable': 'Label'})
elif 'Label' not in df.columns:
    st.error("Neither 'Label' nor 'Lable' column found in the dataset.")
    st.stop()


# ==== Page Title ====

st.markdown("<h1>🔎 Detect Posts</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# ==== General Dataset Information ====

# Calculate dataset statistics:
# - total number of posts
# - suicidal posts (Label = 1)
# - non-suicidal posts (Label = 0)
st.markdown("### 🧾 General Information", unsafe_allow_html=True)
total_posts = len(df)
suicidal = (df['Label'] == 1).sum()
non_suicidal = (df['Label'] == 0).sum()

# Display dataset statistics with colored formatting
st.markdown(
    f"<p style='font-size:18px;'>Total posts: "
    f"<span style='color:green'><b>{total_posts}</b></span></p>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='font-size:18px;'>Suicidal: "
    f"<span style='color:red'><b>{suicidal}</b></span> | Non-Suicidal: "
    f"<span style='color:blue'><b>{non_suicidal}</b></span></p>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)


# ==== Pie Chart: Post Classification ====

# Visualize distribution of suicidal vs non-suicidal posts
st.markdown("### 📊 Post Classification (Suicidal / Non-Suicidal)", unsafe_allow_html=True)

fig1, ax1 = plt.subplots(figsize=(3.8, 3.8))
sizes = [suicidal, non_suicidal]
labels = ['Suicidal', 'Non-Suicidal']
colors = ['#FF6F61', '#6A5ACD']

ax1.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    textprops={'fontsize': 10}
)
ax1.axis('equal')  # Draw as circle
st.pyplot(fig1)

st.markdown("<br>", unsafe_allow_html=True)


# ==== Data Preview Table ====

# Show a preview of the first 10 rows of the dataset
st.markdown("### 👁️ Data Preview (Top 10)", unsafe_allow_html=True)
preview_cols = [c for c in ['Text', 'Label'] if c in df.columns]
st.dataframe(df[preview_cols].head(10), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==== Histogram: Post Length Distribution ====

# Add new column with post length (character count)
st.markdown("### 📏 Post Length Distribution", unsafe_allow_html=True)
df['text_length'] = df['Text'].astype(str).apply(len)

# Plot histogram of post lengths
fig2, ax2 = plt.subplots(figsize=(6, 3.5))
ax2.hist(df['text_length'], bins=50, color='#00BFFF', edgecolor='black')
ax2.set_title('Post Length Distribution')
ax2.set_xlabel('Length (characters)')
ax2.set_ylabel('Number of posts')
ax2.ticklabel_format(axis='y', style='plain')  # Prevent scientific notation
st.pyplot(fig2)
