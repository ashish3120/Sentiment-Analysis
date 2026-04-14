# Sentiment Analysis Production App

A production-ready sentiment analysis application built with Streamlit and Scikit-Learn.

## Project Structure

```text
nlp/
├── src/
│   ├── preprocess.py      # Text cleaning pipeline
│   ├── model.py           # Inference logic (SentimentAnalyzer class)
│   └── __init__.py
├── models/
│   ├── sentiment_model.pkl   # Trained Logistic Regression model
│   ├── tfidf_vectorizer.pkl  # Fitted TF-IDF vectorizer
│   └── index_to_label.pkl    # Label mapping
├── scripts/
│   └── train_model.py     # Script to (re)train the model
├── data/
│   └── train.txt          # Raw training data
├── app.py                 # Streamlit application entry point
├── requirements.txt       # Project dependencies
└── README.md
```

## How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Model (Optional):**
   The model is already trained and saved in `models/`. To retrain:
   ```bash
   python scripts/train_model.py
   ```

3. **Launch the App:**
   ```bash
   streamlit run app.py
   ```

## Model Details
- **Algorithm:** Logistic Regression
- **Vectorization:** TF-IDF
- **Accuracy:** ~86.2% on the validation set.
- **Classes:** sadness, anger, love, surprise, fear, joy.
