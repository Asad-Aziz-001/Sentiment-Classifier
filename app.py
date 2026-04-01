import streamlit as st
import pickle
import os
import base64
from io import BytesIO
import textwrap

# Page configuration (modern look)
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Styles: modern UI with background gradients, glass card, animations ---
st.markdown(
    """
    <style>
    :root{
        --accent-1: #6EE7B7;
        --accent-2: #60A5FA;
        --bg-grad-1: linear-gradient(135deg, rgba(14,165,233,0.10), rgba(99,102,241,0.06));
        --card-grad: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(99,102,241,0.10));
        --glass: rgba(255,255,255,0.06);
        --glass-2: linear-gradient(135deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        --muted: #94a3b8;
    }

    /* Background gradient for the whole page */
    .stApp {
        background: radial-gradient(1200px 600px at 10% 10%, rgba(99,102,241,0.12), transparent 10%),
                    radial-gradient(1000px 500px at 90% 90%, rgba(59,130,246,0.10), transparent 10%),
                    linear-gradient(180deg, #0f172a 0%, #071032 100%);
        color: #e6eef8;
    }

    /* Main container spacing */
    .main {
        padding: 2rem 1.5rem !important;
    }

    /* Glass card */
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border-radius: 18px;
        padding: 1.25rem;
        box-shadow: 0 8px 30px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.04);
    }

    /* Header */
    h1, .css-10trblm { /* heading tweaks for different Streamlit versions */
        letter-spacing: 0.2px;
    }

    .subtitle {
        color: var(--muted);
        margin-top: -0.5rem;
        margin-bottom: 1rem;
    }

    /* Text area styling */
    .stTextArea textarea {
        font-size: 1.05rem;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        background: rgba(2,6,23,0.45) !important;
        color: #e6eef8 !important;
        padding: 0.9rem !important;
        min-height: 160px;
        resize: vertical;
    }

    .stButton>button {
        background: linear-gradient(90deg,#3b82f6,#60a5fa);
        color: white;
        border: none;
        padding: 0.6rem 1rem;
        font-weight: 600;
        border-radius: 10px;
        box-shadow: 0 8px 24px rgba(37,99,235,0.18);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        transition: transform 0.18s ease;
        box-shadow: 0 14px 40px rgba(37,99,235,0.22);
    }

    /* Result card */
    .result-card {
        border-radius: 14px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.04);
    }
    .emoji {
        font-size: 3.8rem;
        margin-bottom: 0.4rem;
    }
    .sentiment-title {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .sentiment-sub {
        color: rgba(255,255,255,0.9);
        opacity: 0.9;
        margin-top: 0.3rem;
    }

    /* small helper text */
    .muted {
        color: var(--muted);
        font-size: 0.95rem;
    }

    /* footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.9rem;
        padding-top: 1rem;
    }

    /* responsive tweaks */
    @media (max-width: 640px) {
        .emoji { font-size: 3rem; }
        .stTextArea textarea { min-height: 140px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header content
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.title("😊 Sentiment Analyzer")
st.markdown("<div class='subtitle'>Analyze the sentiment of your text instantly — modern UI with background gradients</div>", unsafe_allow_html=True)

# Model loader with caching
@st.cache_resource
def load_model():
    try:
        with st.spinner("Loading sentiment model..."):
            # Expecting a tuple (model, vectorizer)
            model_path = "sentiment_model.pkl"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            with open(model_path, "rb") as f:
                model, vectorizer = pickle.load(f)
            return model, vectorizer
    except Exception as e:
        # Propagate the exception for the UI to handle
        raise e

# Attempt to load model and show helpful messages
model = None
vectorizer = None
try:
    model, vectorizer = load_model()
    st.success("✅ Model loaded")
except FileNotFoundError:
    st.error("❌ Model 'sentiment_model.pkl' not found. Place it in the app folder and rerun.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Main input area inside a card
st.markdown("<div style='margin-top:1rem;' class='card'>", unsafe_allow_html=True)
text = st.text_area(
    label="Enter your text here 👇",
    placeholder="Type or paste your review, tweet, comment, or any text...",
    height=180,
    max_chars=4000,
)

# small helper row for quick actions
cols = st.columns([1, 1, 1])
with cols[0]:
    clear = st.button("Clear", use_container_width=True)
with cols[1]:
    copy = st.button("Copy Input")
with cols[2]:
    sample = st.button("Insert Sample")

if clear:
    # clearing the text area is not straightforward; reload with empty value via st.experimental_set_query_params workaround
    st.experimental_set_query_params(_clear="1")
    st.experimental_rerun()

if copy and text.strip() != "":
    # copy to clipboard won't work server-side; provide a small downloadable text file instead
    b = text.encode("utf-8")
    st.download_button("Download Input as .txt", data=b, file_name="input.txt", mime="text/plain")

if sample:
    sample_text = "I absolutely love this product! The quality is great and delivery was fast."
    # Can't programmatically inject into text_area value; re-run with query param for simplicity
    st.experimental_set_query_params(sample=sample_text)
    st.experimental_rerun()

# Support loading text from query param (used above)
params = st.experimental_get_query_params()
if "sample" in params and not text:
    text = params["sample"][0]
if "_clear" in params:
    # reset params and rerun
    st.experimental_set_query_params()
    st.experimental_rerun()

# Analyze button
analyze = st.button("🔍 Analyze Sentiment", help="Click to predict sentiment", key="analyze_btn")

# Helper: compute human-friendly confidence
def get_confidence(model, X):
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            # choose probability of predicted class
            pred_idx = probs.argmax()
            conf = probs[pred_idx]
            return float(conf)
        elif hasattr(model, "decision_function"):
            # scale decision score into 0..1 via logistic
            import numpy as np
            score = model.decision_function(X)[0]
            conf = 1 / (1 + np.exp(-score))
            return float(conf)
        else:
            return None
    except Exception:
        return None

# If user clicks analyze
if analyze:
    if not text or text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            X = vectorizer.transform([text])
            pred = model.predict(X)[0]
            conf = get_confidence(model, X)

            if pred == 1 or str(pred).lower() in ("pos", "positive"):
                sentiment = "Positive"
                emoji = "😊"
                bg = "linear-gradient(135deg, #0f766e, #34d399)"  # teal/green
                accent = "#10b981"
            else:
                sentiment = "Negative"
                emoji = "😞"
                bg = "linear-gradient(135deg, #991b1b, #fb7185)"  # red/pink
                accent = "#ef4444"

            # Render result card
            conf_text = f"{conf*100:.1f}%" if conf is not None else "N/A"
            st.markdown(
                f"""
                <div class="result-card" style="background: {bg};">
                    <div class="emoji">{emoji}</div>
                    <div class="sentiment-title">{sentiment}</div>
                    <div class="sentiment-sub">Your text expresses a <strong>{sentiment.lower()}</strong> sentiment</div>
                    <div style="margin-top:0.6rem; color: rgba(255,255,255,0.9); font-weight:600;">Confidence: {conf_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Two small metrics
            c1, c2 = st.columns(2)
            with c1:
                st.metric(label="Prediction", value=sentiment)
            with c2:
                # If numeric confidence, show as percentage
                if conf is not None:
                    st.metric(label="Confidence", value=f"{conf*100:.1f}%")
                else:
                    st.metric(label="Confidence", value="N/A")

            # Show probabilities if available
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                # Attempt to map classes to probs
                classes = list(model.classes_)
                prob_map = {str(classes[i]): float(probs[i]) for i in range(len(classes))}
                st.write("Class probabilities:")
                st.json(prob_map)

            # Expandable input text
            with st.expander("📝 View Input Text", expanded=False):
                st.write(text)

            # Small actions row: download result summary
            summary = f"Sentiment: {sentiment}\\nConfidence: {conf_text}\\n\\nInput:\\n{text}"
            b = summary.encode("utf-8")
            st.download_button("Download Result", data=b, file_name="sentiment_result.txt", mime="text/plain")
st.markdown("</div>", unsafe_allow_html=True)  # close card

# Footer
st.markdown("<div class='footer'>Built by ASAD AZIZ</div>", unsafe_allow_html=True)
