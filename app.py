import streamlit as st
import pickle
import os
from PIL import Image
import base64

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================== BACKGROUND & CUSTOM CSS ======================
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Use a beautiful gradient + subtle image background (you can change the image)
background_image = """
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e2937 100%);
        background-attachment: fixed;
    }
    
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: url('https://source.unsplash.com/random/1920x1080/?abstract,blue,gradient') no-repeat center center;
        background-size: cover;
        opacity: 0.15;
        z-index: -1;
    }

    .main {
        padding: 2rem 3rem;
    }

    .stTextArea textarea {
        font-size: 1.15rem;
        border-radius: 16px;
        border: 2px solid rgba(59, 130, 246, 0.5);
        background: rgba(15, 23, 42, 0.8);
        color: white;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2);
    }

    .title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a5b4fc, #c4d0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.3rem;
        margin-bottom: 2rem;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2.5rem;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .result-card {
        padding: 3rem 2rem;
        border-radius: 24px;
        text-align: center;
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.4s ease;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    }
    
    .result-card:hover {
        transform: translateY(-10px);
    }

    .emoji {
        font-size: 6rem;
        margin-bottom: 1rem;
        animation: float 3s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-15px); }
    }

    .footer {
        text-align: center;
        color: #64748b;
        margin-top: 4rem;
        font-size: 0.95rem;
    }
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

# ====================== TITLE SECTION ======================
st.markdown('<h1 class="title">🌟 Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Instant AI-powered sentiment analysis with beautiful insights</p>', unsafe_allow_html=True)

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            model, vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model file `sentiment_model.pkl` not found in the app directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model, vectorizer = load_model()

# ====================== INPUT SECTION ======================
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    text = st.text_area(
        label="",
        placeholder="Paste your review, tweet, feedback, or any text here...",
        height=220,
        max_chars=3000,
        label_visibility="collapsed"
    )

# ====================== ANALYZE BUTTON ======================
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_button = st.button(
        "🚀 Analyze Sentiment",
        type="primary",
        use_container_width=True,
        help="Click to get instant sentiment prediction"
    )

# ====================== RESULT SECTION ======================
if analyze_button:
    if not text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🔍 Analyzing your text with AI..."):
            # Prediction
            X = vectorizer.transform([text])
            prediction = model.predict(X)[0]
            
            # Get probability if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                confidence = max(proba) * 100

            # Sentiment settings
            if prediction == 1:
                sentiment = "Positive"
                emoji = "😊"
                gradient = "linear-gradient(135deg, #166534, #4ade80)"
                accent_color = "#4ade80"
            else:
                sentiment = "Negative"
                emoji = "😞"
                gradient = "linear-gradient(135deg, #991b1b, #f87171)"
                accent_color = "#f87171"

            # Display Result
            st.markdown(f"""
                <div class="result-card glass-card" style="background: {gradient};">
                    <div class="emoji">{emoji}</div>
                    <h2 style="margin: 0; font-size: 3rem; color: white; font-weight: 700;">
                        {sentiment}
                    </h2>
                    <p style="margin: 1rem 0 0 0; font-size: 1.35rem; opacity: 0.95;">
                        Your text expresses a <strong>{sentiment.lower()}</strong> sentiment
                    </p>
                </div>
            """, unsafe_allow_html=True)

            # Metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Sentiment", sentiment, delta=None)
            
            with metric_col2:
                st.metric("Confidence", f"{confidence:.1f}%" if confidence else "High", 
                         delta=None)
            
            with metric_col3:
                st.metric("Text Length", f"{len(text)} characters")

            # Input Preview
            with st.expander("📄 View Original Text", expanded=False):
                st.write(text)

# ====================== FOOTER ======================
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        Built with ❤️ by <strong>ASAD AZIZ</strong> • Powered by Machine Learning
    </div>
    """, 
    unsafe_allow_html=True
)

# Optional: Add particles or more effects (advanced)
st.markdown("""
<style>
    .stButton>button {
        height: 3.5rem;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.03);
    }
</style>
""", unsafe_allow_html=True)
