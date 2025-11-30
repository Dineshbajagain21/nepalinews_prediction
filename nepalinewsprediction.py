# File: nepalinewsprediction.py

import pandas as pd
import re
import pickle
from fastapi import FastAPI
import sys


# ============================================
# 1️⃣ Custom Nepali Tokenizer (your real tokenizer)
# ============================================
def nepali_tokenizer(text):
    """
    Custom tokenizer used during model training.
    """
    text = re.sub(r"[^अ-ह०-९\s]", " ", text)
    tokens = text.split()

    nepali_stopwords = ["र", "का", "की", "ले", "छ", "था", "देखि", "मा", "को"]
    tokens = [t for t in tokens if t not in nepali_stopwords]

    return tokens


# ============================================
# 2️⃣ FIX — Register tokenizer name for pickle
# ============================================
# This allows pickle to find the tokenizer during model loading
sys.modules['__main__'].nepali_tokenizer = nepali_tokenizer


# ============================================
# 3️⃣ Load your trained model safely
# ============================================
with open("Nepalinewsregression2.pickle", "rb") as f:
    model = pickle.load(f)


# ============================================
# 4️⃣ FastAPI app
# ============================================
app = FastAPI(
    title="Nepali News Classifier",
    description="Enter Nepali news text and get predicted category",
    version="1.0"
)


# ============================================
# 5️⃣ News categories
# ============================================
categories = {
    0: "politics",
    1: "entertainment",
    2: "finance",
    3: "film",
    4: "science_technology",
    5: "literature",
    6: "society",
    7: "opinion",
    8: "tourism",
    9: "national",
    10: "migration",
    11: "corporate",
    12: "accidents",
    13: "sports",
    14: "wealth",
    15: "health",
    16: "auto",
    17: "interview"
}


# ============================================
# 6️⃣ Prediction endpoint
# ============================================
@app.post("/predict_news")
def predict_news(news_text: str):
    df = pd.DataFrame({'predict_news': [news_text]})
    prediction = model.predict(df['predict_news'])[0]
    
    # Also return the raw prediction to debug
    return {
        "input_text": news_text,
        "raw_model_output": prediction
    }


# ============================================
# 7️⃣ Root endpoint
# ============================================
@app.get("/")
def root():
    return {"message": "Welcome to Nepali News Classifier API. Use POST /predict_news"}
