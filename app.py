import streamlit as st
import pickle

st.set_page_config(page_title="EmotionSense", page_icon="🧠", layout="centered")

# 🔥 CSS (converted from your HTML)
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
}

/* Card */
.card {
    background: rgba(255,255,255,0.1);
    padding: 2rem;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

/* Title */
.title {
    font-size: 2.5rem;
    font-weight: bold;
    background: linear-gradient(to right, #fff, #ddd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}

/* Textarea */
textarea {
    border-radius: 12px !important;
    background: rgba(255,255,255,0.1) !important;
    color: white !important;
}

/* Button */
.stButton>button {
    border-radius: 10px;
    background: linear-gradient(135deg, #ff6b81, #ff4757);
    color: white;
    font-size: 1rem;
}

/* Result */
.result {
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    margin-top: 1rem;
}

</style>
""", unsafe_allow_html=True)

# 🔥 Title
st.markdown("<div class='title'>🧠 EmotionSense</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI Sentiment Analysis Tool</p>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model, vectorizer = pickle.load(open('sentiment_model.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_model()

# Session state (history + stats)
if "history" not in st.session_state:
    st.session_state.history = []

if "stats" not in st.session_state:
    st.session_state.stats = {"total":0, "positive":0, "negative":0}

# Main card
st.markdown("<div class='card'>", unsafe_allow_html=True)

text = st.text_area("Enter text", label_visibility="collapsed", height=150)

col1, col2 = st.columns(2)

with col1:
    analyze = st.button("🚀 Analyze")

with col2:
    clear = st.button("🧹 Clear")

# Clear
if clear:
    st.session_state.history = []
    st.session_state.stats = {"total":0, "positive":0, "negative":0}
    st.rerun()

# Analyze
if analyze:
    if text.strip() == "":
        st.warning("Enter some text")
    else:
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]

        if prediction == 1:
            sentiment = "Positive"
            emoji = "😎"
            color = "#22c55e"
        else:
            sentiment = "Negative"
            emoji = "😡"
            color = "#ef4444"

        # Save history
        st.session_state.history.insert(0, (text, sentiment))
        st.session_state.history = st.session_state.history[:10]

        # Stats update
        st.session_state.stats["total"] += 1
        if sentiment == "Positive":
            st.session_state.stats["positive"] += 1
        else:
            st.session_state.stats["negative"] += 1

        # Result UI
        st.markdown(f"""
        <div class="result" style="background:{color};">
            <h1>{emoji}</h1>
            <h2>{sentiment}</h2>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# 📊 Stats
st.markdown("### 📊 Statistics")
col1, col2, col3 = st.columns(3)

col1.metric("Total", st.session_state.stats["total"])
col2.metric("Positive", st.session_state.stats["positive"])
col3.metric("Negative", st.session_state.stats["negative"])

# 🕒 History
st.markdown("### 🕒 History")

for item in st.session_state.history:
    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.1);
        padding:10px;
        border-radius:10px;
        margin-bottom:8px;">
        <b>{item[1]}</b> — {item[0][:50]}
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown(
    "<p style='text-align:center; opacity:0.7;'>Built by Asad Aziz 🚀</p>",
    unsafe_allow_html=True
)
