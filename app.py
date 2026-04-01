import streamlit as st
import pickle

st.set_page_config(page_title="Sentiment AI", page_icon="🚀", layout="centered")

# 🌈 Advanced CSS
st.markdown("""
<style>

/* Animated Gradient Background */
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e3a8a, #2563eb, #38bdf8);
    background-size: 400% 400%;
    animation: gradient 12s ease infinite;
    color: white;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass Card */
.glass {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
    padding: 2rem;
    border-radius: 25px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}

/* Title Glow */
.title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: bold;
    background: linear-gradient(90deg, #38bdf8, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Textarea */
textarea {
    border-radius: 15px !important;
    border: 2px solid #60a5fa !important;
    background: rgba(255,255,255,0.1) !important;
    color: white !important;
}

/* Button */
.stButton>button {
    border-radius: 15px;
    background: linear-gradient(135deg, #3b82f6, #22c55e);
    color: white;
    font-size: 1.1rem;
    padding: 0.7rem;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.07);
    box-shadow: 0 0 20px #3b82f6;
}

/* Result Card Glow */
.result {
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    margin-top: 1.5rem;
    animation: fadeIn 0.5s ease-in-out;
}

@keyframes fadeIn {
    from {opacity:0; transform: translateY(20px);}
    to {opacity:1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)

# 🔥 Title
st.markdown("<div class='title'>🚀 Sentiment AI Analyzer</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Premium AI-powered sentiment detection</p>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model, vectorizer = pickle.load(open('sentiment_model.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_model()

# Glass Container
st.markdown("<div class='glass'>", unsafe_allow_html=True)

text = st.text_area("Enter your text 👇", height=150)

if st.button("✨ Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Enter some text first")
    else:
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]

        if prediction == 1:
            sentiment = "Positive"
            emoji = "😎"
            bg = "linear-gradient(135deg, #16a34a, #4ade80)"
        else:
            sentiment = "Negative"
            emoji = "😡"
            bg = "linear-gradient(135deg, #dc2626, #f87171)"

        st.markdown(f"""
        <div class="result" style="background:{bg};">
            <h1 style="font-size:4rem;">{emoji}</h1>
            <h2>{sentiment}</h2>
            <p>AI detected a {sentiment.lower()} sentiment</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    "<p style='text-align:center; opacity:0.6; margin-top:20px;'>Built by Asad Aziz ⚡</p>",
    unsafe_allow_html=True
)
