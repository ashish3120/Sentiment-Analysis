---
title: Sentiment Analysis AI
emoji: 🎭
colorFrom: green
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# 🎭 Sentiment Analysis AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade NLP application that identifies six core human emotions with high precision. Built with **Streamlit** and **Scikit-Learn**, this tool provides a sleek interface for real-time emotional intelligence.

---

## ✨ Key Features
- **🚀 Real-time Inference**: Instance sentiment detection as you type.
- **📊 Detailed Metrics**: Shows prediction confidence percentages.
- **🛠️ Modular Architecture**: Production-ready folder structure with separated preprocessing and model logic.
- **🧠 Advanced ML**: Uses Logistic Regression and TF-IDF vectorization for optimal performance (~86% accuracy).

---

## 🛠️ Technology Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/)
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **NLP Utilities**: [NLTK](https://www.nltk.org/)
- **Serialization**: [Joblib](https://joblib.readthedocs.io/)

---

## 📁 Project Structure
```text
Sentiment-Analysis/
├── 📂 src/                # Core Logic
│   ├── preprocess.py      # Text cleaning pipeline
│   └── model.py           # Inference class (SentimentAnalyzer)
├── 📂 models/             # Trained Artifacts
│   ├── sentiment_model.pkl   # Logistic Regression Model
│   └── tfidf_vectorizer.pkl  # TF-IDF Vectorizer
├── 📂 scripts/            # Automation
│   └── train_model.py     # Reproducible training script
├── 📂 data/               # Training Datasets
├── app.py                 # Streamlit UI
├── requirements.txt       # Dependencies
└── README.md              # Project Documentation
```

---

## 🚀 Getting Started

### 1. Prerequisite
Ensure you have Python 3.10+ installed.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/ashish3120/Sentiment-Analysis.git
cd Sentiment-Analysis

# Install dependencies
pip install -r requirements.txt
```

### 3. Usage
```bash
streamlit run app.py
```

---

## 🎯 Model Overview
The model is trained to recognize the following emotions:
- 😢 **Sadness**
- 😡 **Anger**
- 🥰 **Love**
- 😮 **Surprise**
- 😨 **Fear**
- 😊 **Joy**

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.

---
<p align="center">Built with ❤️ for better emotional understanding.</p>
