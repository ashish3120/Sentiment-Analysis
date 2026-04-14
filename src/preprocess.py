import string
import nltk
from nltk.corpus import stopwords

# Ensure NLTK data is downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception:
    pass

STOP_WORDS = set(stopwords.words('english'))

def remove_punc(text: str) -> str:
    """Remove punctuation from text."""
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text: str) -> str:
    """Remove digits from text."""
    return ''.join([i for i in text if not i.isdigit()])

def remove_emojis(text: str) -> str:
    """Remove non-ASCII characters (emojis)."""
    return text.encode('ascii', 'ignore').decode('ascii')

def remove_stopwords(text: str) -> str:
    """Remove English stopwords."""
    words = text.split()
    cleaned = [w for w in words if w.lower() not in STOP_WORDS]
    return ' '.join(cleaned)

def clean_text(text: str) -> str:
    """Complete text cleaning pipeline."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = remove_punc(text)
    text = remove_numbers(text)
    text = remove_emojis(text)
    text = remove_stopwords(text)
    return text
