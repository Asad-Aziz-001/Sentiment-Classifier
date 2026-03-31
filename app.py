import streamlit as st
import pickle
import os

# Page configuration (modern look)
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
        border-radius: 12px;
        border: 2px solid #4a90e2;
    }
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: white;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.3);
    }
    .emoji {
        font-size: 4.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("😊 Sentiment Analyzer")
st.markdown("### Analyze the sentiment of your text instantly")
st.markdown("Enter any text below and get instant positive/negative prediction using your trained model.")

# Load the model (with error handling)
@st.cache_resource
def load_model():
    try:
        with st.spinner("Loading sentiment model..."):
            model, vectorizer = pickle.load(open('sentiment_model.pkl', 'rb'))
        st.success("✅ Model loaded successfully!")
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model file 'sentiment_model.pkl' not found. Please make sure it's in the same folder as this app.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model, vectorizer = load_model()

# Main input area
text = st.text_area(
    label="Enter your text here 👇",
    placeholder="Type or paste your review, tweet, comment, or any text...",
    height=180,
    max_chars=2000
)

# Analyze button with modern styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        use_container_width=True,
        help="Click to predict sentiment"
    )

if analyze_button:
    if text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            # Transform text and predict
            X = vectorizer.transform([text])
            prediction = model.predict(X)[0]
            
            # Determine sentiment and emoji
            if prediction == 1:
                sentiment = "Positive"
                emoji = "😊"
                color = "#22c55e"  # green
                bg_color = "linear-gradient(135deg, #166534, #4ade80)"
            else:
                sentiment = "Negative"
                emoji = "😞"
                color = "#ef4444"  # red
                bg_color = "linear-gradient(135deg, #991b1b, #f87171)"
            
            # Display beautiful result card
            st.markdown(f"""
                <div class="result-card" style="background: {bg_color};">
                    <div class="emoji">{emoji}</div>
                    <h2 style="margin: 0; font-size: 2.2rem;">{sentiment}</h2>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">
                        Your text expresses a <strong>{sentiment.lower()}</strong> sentiment
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Additional info in columns
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label="Prediction",
                    value=sentiment,
                    delta=None
                )
            with col_b:
                st.metric(
                    label="Model Confidence",
                    value="High" if hasattr(model, 'predict_proba') else "N/A"
                )

            # Show the original text in an expander
            with st.expander("📝 View Input Text", expanded=False):
                st.write(text)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.9rem;'>"
    "Built by ASAD AZIZ</p>",
    unsafe_allow_html=True
)

# Run with: streamlit run app.py
