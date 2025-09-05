import streamlit as st

# ==== Page Configuration ====

st.set_page_config(page_title="About Us 💜", layout="wide")


# ==== Custom CSS Styling ====

st.markdown("""
<style>
.stMarkdown, .stText, .stTitle, .stSubtitle, .stCaption, .stMarkdownContainer {
    direction: ltr;
    text-align: left;
    font-family: "Segoe UI", sans-serif;
    color: #1d1d1d;
}
.transparent-box {
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 30px;
    margin: auto;
    width: 70%;
    text-align: left;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}
h3 { color: #003366; }
</style>
""", unsafe_allow_html=True)


# ==== Content Container ====

st.markdown('<div class="transparent-box">', unsafe_allow_html=True)


# ==== Who Are We? ====

st.markdown("## Who Are We? 💜", unsafe_allow_html=True)
st.markdown(
    """
<p style="font-size: 18px">
We are Information Systems students at Yezreel Valley College, and our project focuses on the early detection of signs of mental distress on social media.
We chose this topic because it combines a technological challenge with a meaningful human contribution — using Artificial Intelligence to help in suicide prevention.
</p>
""",
    unsafe_allow_html=True,
)


# ==== Background & Importance ====

st.markdown('<h3 style="margin-top: 40px;">📌 Background & Importance:</h3>', unsafe_allow_html=True)
st.markdown(
    """
<p style="font-size: 18px">
Suicide is a sensitive and urgent issue, with one person dying by suicide every 40 seconds worldwide.<br>
Many people share their struggles on social media.<br>
We developed an Artificial Intelligence model that helps identify signs of distress and potentially save lives.
</p>
""",
    unsafe_allow_html=True,
)


# ==== Project Goals ====

st.markdown('<h3 style="margin-top: 40px;">🎯 Project Goals:</h3>', unsafe_allow_html=True)
st.markdown(
    """
<ul style="font-size: 18px">
<li>Identify textual signals that indicate psychological distress accurately and reliably.</li>
<li>Build an Artificial Intelligence (AI) based model.</li>
<li>Apply Natural Language Processing (NLP) techniques.</li>
<li>Train multiple models and evaluate which one performs best.</li>
<li>Develop a simple, user-friendly interface that can serve as a tool for detecting distress.</li>
</ul>
""",
    unsafe_allow_html=True,
)


# ==== End of Container ====

st.markdown("</div>", unsafe_allow_html=True)
