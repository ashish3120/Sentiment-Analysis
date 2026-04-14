import pandas as pd
import joblib
import os
import sys

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train():
    print("Loading data...")
    # Load training data
    train_path = os.path.join('data', 'train.txt')
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    df = pd.read_csv(train_path, sep=';', header=None, names=['text', 'emotion'])
    
    # Create emotion mapping
    unique_emotions = df['emotion'].unique()
    label_to_index = {label: i for i, label in enumerate(unique_emotions)}
    index_to_label = {i: label for label, i in label_to_index.items()}
    
    print(f"Emotion mapping: {label_to_index}")
    df['label'] = df['emotion'].map(label_to_index)
    
    print("Cleaning text...")
    df['text_cleaned'] = df['text'].apply(clean_text)
    
    print("Vectorizing...")
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['text_cleaned'])
    y = df['label']
    
    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Save artifacts
    print("Saving artifacts...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, os.path.join('models', 'sentiment_model.pkl'))
    joblib.dump(tfidf, os.path.join('models', 'tfidf_vectorizer.pkl'))
    joblib.dump(index_to_label, os.path.join('models', 'index_to_label.pkl'))
    
    print("Training complete. Artifacts saved in 'models/'")

if __name__ == "__main__":
    train()
