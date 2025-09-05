import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from pathlib import Path


# ==== Page Configuration ====

# Use a wide layout for side-by-side charts.
st.set_page_config(layout="wide")


# ==== Custom CSS Styling ====

# Left-to-right page with a light background and left-aligned text.
st.markdown("""
<style>
    body { background-color: #f7fbff; direction: ltr; }
    .main { background-color: #f7fbff; }
    h1, h2, h3, h4, h5, h6, p { text-align: left; direction: ltr; }
</style>
""", unsafe_allow_html=True)


# ==== Paths (Portable ROOT) ====

# Determine project root from this file (pages/ -> dashboard/ -> src/ -> ROOT).
ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = ROOT / "data" / "processed" / "final_balanced_dataset.csv"

# Safety check before loading.
if not DATASET_PATH.is_file():
    st.error(f"Dataset not found at: {DATASET_PATH}")
    st.stop()


# ==== Load & Prepare Data ====

# Load the dataset.
df = pd.read_csv(DATASET_PATH)

# Normalize label column name (support both 'Lable' and 'Label').
if 'Lable' in df.columns and 'Label' not in df.columns:
    df = df.rename(columns={'Lable': 'Label'})
elif 'Label' not in df.columns:
    st.error("Neither 'Label' nor 'Lable' column exists in the dataset.")
    st.stop()

# Ensure label is integer and text is string.
df['Label'] = df['Label'].astype(int)
df['Text'] = df['Text'].astype(str)

# Split by class.
suicidal_posts = df[df['Label'] == 1]
nonsuicidal_posts = df[df['Label'] == 0]


# ==== Page Title ====

st.markdown("<h2>🔍 Statistical Analysis of Posts</h2>", unsafe_allow_html=True)


# ==== Word Clouds (Per Class) ====

def generate_wordcloud(text_series: pd.Series, title: str):
    """
    Generate and return a matplotlib figure with a word cloud.
    """
    # Join all posts into one text; WordCloud handles tokenization & frequency.
    joined = " ".join(text_series.tolist())
    wordcloud = WordCloud(width=600, height=300, background_color="white").generate(joined)

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    return fig


col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔴 Suicidal Posts")
    fig1 = generate_wordcloud(suicidal_posts['Text'], "Suicidal Posts")
    st.pyplot(fig1)

with col2:
    st.markdown("#### 🟢 Non-Suicidal Posts")
    fig2 = generate_wordcloud(nonsuicidal_posts['Text'], "Non-Suicidal Posts")
    st.pyplot(fig2)


# ==== Separator ====

st.markdown("<hr>", unsafe_allow_html=True)


# ==== Average Post Length by Label ====

# Compute token-based length (word count) for each post.
df['post_length'] = df['Text'].apply(lambda x: len(x.split()))

avg_length = (
    df.groupby('Label', as_index=False)['post_length']
      .mean()
      .rename(columns={'post_length': 'Average_Length'})
)

# Map numeric labels to readable names.
label_map = {0: 'Non-Suicidal', 1: 'Suicidal'}
avg_length['Label'] = avg_length['Label'].map(label_map)

st.markdown("#### ✏️ Average Post Length by Label")

# Small, compact bar chart.
fig3, ax3 = plt.subplots(figsize=(3, 1.5))
sns.barplot(data=avg_length, x='Label', y='Average_Length', ax=ax3, palette=['tab:orange', 'tab:blue'])
ax3.set_ylabel("Average Length (words)")
ax3.set_xlabel("")
ax3.set_title("Average Post Length by Label", fontsize=9)
st.pyplot(fig3)


# ==== Separator ====

st.markdown("<hr>", unsafe_allow_html=True)


# ==== Top Word Frequency Comparison (Per Class) ====

st.markdown("#### 📊 Top Word Frequency Comparison")

def get_top_words(corpus, n=15):
    """
    Return the top-n words and their frequencies for a given corpus (list/Series of texts),
    using English stop-words removal.
    """
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, int(sum_words[0, idx])) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

TOP_N = 15
suicidal_words = dict(get_top_words(suicidal_posts['Text'], TOP_N))
nonsuicidal_words = dict(get_top_words(nonsuicidal_posts['Text'], TOP_N))

# Combine unique words from both groups.
all_words = sorted(set(list(suicidal_words.keys()) + list(nonsuicidal_words.keys())))
suicidal_freq = [suicidal_words.get(word, 0) for word in all_words]
nonsuicidal_freq = [nonsuicidal_words.get(word, 0) for word in all_words]

freq_df = pd.DataFrame({
    'Word': all_words,
    'Suicidal_Freq': suicidal_freq,
    'NonSuicidal_Freq': nonsuicidal_freq
})

# Compact grouped bar chart.
fig4, ax4 = plt.subplots(figsize=(6, 3))
bar_width = 0.35
index = range(len(freq_df))

ax4.bar(index, freq_df['Suicidal_Freq'], bar_width, label='Suicidal', color='tab:blue')
ax4.bar([i + bar_width for i in index], freq_df['NonSuicidal_Freq'], bar_width, label='Non-Suicidal', color='tab:orange')

ax4.set_xlabel('Word')
ax4.set_ylabel('Frequency')
ax4.set_title('Top Word Frequencies Comparison', fontsize=12)
ax4.set_xticks([i + bar_width / 2 for i in index])
ax4.set_xticklabels(freq_df['Word'], rotation=45, ha='right')
ax4.legend()

st.pyplot(fig4)

# Support table (sorted by suicidal frequency by default).
st.markdown("##### 🧾 Word Frequency Table")
st.dataframe(freq_df.sort_values(by="Suicidal_Freq", ascending=False).reset_index(drop=True), use_container_width=True)
