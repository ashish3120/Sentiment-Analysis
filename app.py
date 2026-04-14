import streamlit as st
import pandas as pd
import time
import os
from src.model import SentimentAnalyzer

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .sentiment-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .sentiment-label {
        font-size: 24px;
        font-weight: bold;
        text-transform: uppercase;
    }
    .confidence-score {
        font-size: 18px;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Sidebar
    with st.sidebar:
        st.title("Settings")
        st.info("This AI analyzes the emotional sentiment of your text using a Logistic Regression model trained on 16k labeled examples.")
        st.write("---")
        st.write("### Model Info")
        st.write("- **Algorithm:** Logistic Regression")
        st.write("- **Features:** TF-IDF Vectorization")
        st.write("- **Classes:** Sadness, Anger, Love, Surprise, Fear, Joy")

    # Header
    st.title("🎭 Sentiment Analysis AI")
    st.subheader("Understand the emotions behind the text")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area("Enter your text below:", height=200, placeholder="Type something emotional here...")
        
        if st.button("Analyze Sentiment"):
            if user_input.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    label, confidence = analyzer.predict(user_input)
                    time.sleep(0.5) # For effect
                    
                    # Display results in columns
                    st.write("### Prediction Result")
                    
                    # Map labels to icons
                    icons = {
                        "sadness": "😢",
                        "anger": "😡",
                        "love": "🥰",
                        "surprise": "😮",
                        "fear": "😨",
                        "joy": "😊"
                    }
                    icon = icons.get(label, "❓")
                    
                    st.markdown(f"""
                        <div class="sentiment-card">
                            <div class="sentiment-label">{icon} {label}</div>
                            <div class="confidence-score">Confidence Score: {confidence:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.success(f"Calculated predicted emotion as **{label}** with **{confidence:.2%}** confidence.")

    with col2:
        st.write("### Examples")
        examples = [
            "I feel like I'm finally reaching my goals and everything is falling into place.",
            "I am so incredibly angry that they lied to me about the deadline.",
            "I'm feeling a bit overwhelmed and scared about the upcoming changes.",
            "I am surprised to see how much progress we have made in such a short time."
        ]
        selected_example = st.selectbox("Choose an example to test:", ["None"] + examples)
        if selected_example != "None":
            st.info(f"Selected: {selected_example}")
            # Highlight this as something they can copy-paste

    # Footer
    st.write("---")
    st.caption("Built with  by Antigravity AI | Production Ready Structure")

if __name__ == "__main__":
    main()
