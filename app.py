import streamlit as st
import pickle

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

/* Glass Card */
.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Title */
.title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #cbd5f5;
    margin-bottom: 1.5rem;
}

/* Text Area */
textarea {
    border-radius: 12px !important;
    border: 2px solid #6366f1 !important;
    background-color: #0f172a !important;
    color: white !important;
}

/* Button */
.stButton button {
    border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-size: 1.1rem;
    padding: 0.6rem;
    border: none;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(99,102,241,0.6);
}

/* Result Card */
.result {
    text-align: center;
    padding: 2rem;
    border-radius: 20px;
    margin-top: 1.5rem;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="title">🧠 AI Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze emotions in text with a sleek modern interface</div>', unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    try:
        model, vectorizer = pickle.load(open('sentiment_model.pkl', 'rb'))
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, vectorizer = load_model()

# ------------------ INPUT CARD ------------------
with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    text = st.text_area(
        "Enter your text",
        placeholder="Type a review, tweet, or sentence...",
        height=150
    )

    analyze = st.button("🚀 Analyze Sentiment", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ PREDICTION ------------------
if analyze:
    if not text.strip():
        st.warning("Please enter text")
    else:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]

        if pred == 1:
            sentiment = "Positive"
            emoji = "😊"
            bg = "linear-gradient(135deg, #16a34a, #4ade80)"
        else:
            sentiment = "Negative"
            emoji = "😞"
            bg = "linear-gradient(135deg, #dc2626, #f87171)"

        st.markdown(f"""
        <div class="result" style="background: {bg};">
            <h1 style="font-size:4rem;">{emoji}</h1>
            <h2>{sentiment}</h2>
        </div>
        """, unsafe_allow_html=True)

        # Confidence
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0]
            confidence = max(prob) * 100
            st.progress(int(confidence))
            st.write(f"Confidence: {confidence:.2f}%")

        # Expand text
        with st.expander("View Input Text"):
            st.write(text)

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Built by ASAD AZIZ 🚀</p>", unsafe_allow_html=True)
