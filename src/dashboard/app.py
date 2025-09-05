import streamlit as st
import base64
from pathlib import Path


# ==== Page Configuration ====

st.set_page_config(
    page_title="Mental Distress Detection System",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ==== Background Utilities ====

def get_base64_of_bin_file(bin_file: Path):
    """
    Convert a binary file (e.g., image) into a base64 string
    so it can be used as a CSS background.
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(image_file: Path):
    """
    Apply a background image to the Streamlit app.
    """
    bin_str = get_base64_of_bin_file(image_file)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ==== Set Background (Portable with ROOT) ====

# ROOT -> project root
ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = ROOT / "src" / "dashboard" / "assets" / "image.jpg"

if IMAGE_PATH.is_file():
    set_background(IMAGE_PATH)
else:
    st.warning(f"Background image not found at: {IMAGE_PATH}")


# ==== Top Navigation ====

st.markdown(
    """
    <div style='background-color: rgba(255, 255, 255, 0.85); padding: 12px; border-radius: 12px; text-align: center; margin-bottom: 30px;'>
        <a href="/" style="margin: 0 15px; font-size: 18px; font-weight: bold; text-decoration: none; color: #004080;'>🏠 Home</a>
        <a href="/1_about" style="margin: 0 15px; font-size: 18px; font-weight: bold; text-decoration: none; color: #004080;'>📌 About</a>
        <a href="/2_detect_posts" style="margin: 0 15px; font-size: 18px; font-weight: bold; text-decoration: none; color: #004080;'>🔎 Detect Posts</a>
        <a href="/3_stats_analysis" style="margin: 0 15px; font-size: 18px; font-weight: bold; text-decoration: none; color: #004080;'>📊 Stats Analysis</a>
        <a href="/4_predict_post" style="margin: 0 15px; font-size: 18px; font-weight: bold; text-decoration: none; color: #004080;'>🧠 Predict Post</a>
    </div>
    """, unsafe_allow_html=True
)


# ==== Main Content ====

st.markdown("""
<div style='background-color: rgba(255,255,255, 0.85); padding: 30px; border-radius: 20px; text-align: center;'>
    <h1 style='color: #002060;'>Suicidal Behavior Detection System</h1>
    <p style='font-size: 20px;'>
        This project is designed to identify and classify social media posts that indicate suicidal intent.<br>
        The system leverages advanced text analysis and machine learning models to distinguish between suicidal and non-suicidal posts, helping to support early detection and timely intervention.
    </p>
</div>
""", unsafe_allow_html=True)
