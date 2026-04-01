import streamlit as st
import pickle

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="✨",
    layout="centered"
)

# 🔥 FULL PAGE GRADIENT + GLASS UI
st.markdown("""
<style>
/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e3a8a, #2563eb);
    color: white;
}

/* Glass Card */
.glass {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(15px);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Text Area */
textarea {
    border-radius: 12px !important;
    border: 2px solid #60a5fa !important;
    background: rgba(255,255,255,0.1) !important;
    color: white !important;
}

/* Button */
.stButton>button {
    border-radius: 12px;
    background: linear-gradient(135deg, #3b82f6, #60a5fa);
    color: white;
    font-size: 1rem;
    padding: 0.6rem;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}

/* Result Card */
.result {
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center;'>✨ Sentiment Analyzer</h1>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model, vectorizer = pickle.load(open('sentiment_model.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_model()

# Glass container
with st.container():
    st.markdown("<p style='text-align:center;'>Analyze your text with AI 🚀</p>", unsafe_allow_html=True)

    text = st.text_area("Enter text 👇", height=150)

    if st.button("🔍 Analyze"):
        if text.strip() == "":
            st.warning("Enter some text")
        else:
            X = vectorizer.transform([text])
            prediction = model.predict(X)[0]

            if prediction == 1:
                sentiment = "Positive"
                emoji = "😊"
                bg = "linear-gradient(135deg, #16a34a, #4ade80)"
            else:
                sentiment = "Negative"
                emoji = "😞"
                bg = "linear-gradient(135deg, #dc2626, #f87171)"

            st.markdown(f"""
            <div class="result" style="background:{bg};">
                <h1>{emoji}</h1>
                <h2>{sentiment}</h2>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    "<p style='text-align:center; opacity:0.7;'>Built by Asad Aziz</p>",
    unsafe_allow_html=True
)
