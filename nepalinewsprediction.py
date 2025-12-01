import streamlit as st
import pickle
import re
import sys
import numpy as np
# -----------------------------
# Tokenizer (same as training)
# -----------------------------
def nepali_tokenizer(text):
    text = re.sub(r"[^अ-ह०-९\s]", " ", text)
    tokens = text.split()
    stopwords = ["र", "का", "की", "ले", "छ", "था", "देखि", "मा", "को"]
    return [t for t in tokens if t not in stopwords]

sys.modules['__main__'].nepali_tokenizer = nepali_tokenizer

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "Nepalinewsregression2.pickle"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📰 Nepali News Classification (Raw Output)")
st.write("Enter Nepali news and see the model's raw predicted category.")

news_text = st.text_area("Enter Nepali news:")

if st.button("🔍 Predict Category"):
    if not news_text.strip():
        st.warning("Please enter some text!")
    else:
        try:
            # Predict raw output
            raw_pred = model.predict([news_text])
            
            # If output is array/list, get first element
            predicted_category = raw_pred[0] if isinstance(raw_pred, (list, tuple, np.ndarray)) else raw_pred

            st.success(f"📌 Predicted Category: **{predicted_category}**")
            st.write("🔧 Raw model output:", raw_pred)

        except Exception as e:
            st.error(f"Prediction error: {e}")
