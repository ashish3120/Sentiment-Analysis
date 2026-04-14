import joblib
import os
import sys

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import clean_text

class SentimentAnalyzer:
    def __init__(self, models_dir='models'):
        self.model_path = os.path.join(models_dir, 'sentiment_model.pkl')
        self.vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
        self.label_map_path = os.path.join(models_dir, 'index_to_label.pkl')
        
        self.model = None
        self.vectorizer = None
        self.index_to_label = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load trained models and vectorizers."""
        if all(os.path.exists(p) for p in [self.model_path, self.vectorizer_path, self.label_map_path]):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.index_to_label = joblib.load(self.label_map_path)
        else:
            print("Warning: Model artifacts not found. Please run scripts/train_model.py first.")

    def predict(self, text: str):
        """Predict sentiment for a given text."""
        if self.model is None:
            return None, 0.0
            
        cleaned = clean_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        
        prediction_idx = self.model.predict(vectorized)[0]
        probabilities = self.model.predict_proba(vectorized)[0]
        
        label = self.index_to_label[prediction_idx]
        confidence = probabilities[prediction_idx]
        
        return label, confidence
