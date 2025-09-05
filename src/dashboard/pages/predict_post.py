import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import torch


# ==== Page Configuration ====

# Set page title, icon, and layout
st.set_page_config(page_title="Predict Post", page_icon="🔍", layout="wide")


# ==== Custom CSS Styling ====

# Light background, left-to-right orientation, left-aligned text
st.markdown("""
    <style>
        body { background-color: #f7fbff; direction: ltr; }
        .main { background-color: #f7fbff; }
        h1, h2, h3, h4, h5, h6, p, label { text-align: left; direction: ltr; }
    </style>
""", unsafe_allow_html=True)


# ==== Page Title ====

st.title("🔍 Post Analysis")


# ==== ROOT & Paths (Portable) ====

# Project root: go up two levels from this file (pages/ -> dashboard/ -> src/ -> ROOT)
ROOT = Path(__file__).resolve().parents[2]

# Path to the locally fine-tuned model (adjust if your folder name differs)
MODEL_DIR = ROOT / "modeling" / "final" / "DistilBERT" / "distilbert_final_model"

# ==== Utilities: Model Loading & Single Prediction ====

@st.cache_resource(show_spinner=False)
def load_local_model(model_dir: Path):
    """
    Load tokenizer and model from a local directory (no internet).
    Cached so it loads only once per session.
    """
    # Basic validations to avoid confusing runtime errors
    if not model_dir.is_dir():
        st.error(f"Model directory was not found:\n{model_dir}")
        st.stop()

    # Try loading local artifacts only
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    except Exception as e:
        st.error(
            "Failed to load the local model/tokenizer. "
            "Make sure the folder contains the fine-tuned model files."
        )
        st.exception(e)
        st.stop()

    # Put model in eval mode on the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device


def predict_single(tokenizer, model, device, text: str, max_len: int = 256):
    """
    Predict a single text. Returns:
    - pred_label (int): 0 = Non-Suicidal, 1 = Suicidal
    - pred_prob (float): probability of the predicted class (0..1)
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1).squeeze(0)  # shape [2]
        pred_label = int(torch.argmax(probs).item())
        pred_prob = float(torch.max(probs).item())

    return pred_label, pred_prob


# ==== Load Model Once ====

tokenizer, model, device = load_local_model(MODEL_DIR)


# ==== User Input ====

user_input = st.text_area("✍ Enter the post content", height=200, placeholder="Paste or type the post text here...")


# ==== Run Prediction ====

if st.button("🔎 Run Prediction"):
    if not user_input.strip():
        st.warning("Please enter some text before running the prediction.")
    else:
        with st.spinner("Running prediction..."):
            label, prob = predict_single(tokenizer, model, device, user_input)

        # Map label to friendly text
        label_text = "Suicidal" if label == 1 else "Non-Suicidal"
        prob_pct = round(prob * 100, 2)

        # Show result with visual emphasis
        if label == 1:
            st.error(f"🛑 The post is detected as **{label_text}** (confidence: {prob_pct}%).")
        else:
            st.success(f"✅ The post is detected as **{label_text}** (confidence: {prob_pct}%).")

        # Optional: show raw details for debugging / transparency
        with st.expander("Show details"):
            st.write({
                "predicted_label": label,
                "predicted_label_text": label_text,
                "predicted_confidence": prob,
                "model_path": str(MODEL_DIR),
                "device": str(device),
            })
