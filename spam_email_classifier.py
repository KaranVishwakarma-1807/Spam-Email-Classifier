# spam_email_classifier project with Streamlit frontend

# 1. Data Loading and Preprocessing
import pandas as pd
import re
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import os

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load or train model and vectorizer
MODEL_PATH = "models/spam_classifier.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        df = pd.read_csv("data/spam.csv", encoding='latin-1')[["v1", "v2"]]
        df.columns = ["label", "message"]
        df["clean_message"] = df["message"].apply(clean_text)

        vectorizer = TfidfVectorizer(max_features=3000)
        X = vectorizer.fit_transform(df["clean_message"])
        y = df["label"].map({"ham": 0, "spam": 1})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MultinomialNB()
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_model()

def predict_email(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    return "Spam" if model.predict(vector)[0] == 1 else "Not Spam"

# Streamlit app
st.title("📧 Spam Email Classifier")
st.write("Enter an email message below to determine if it's spam or not.")

input_text = st.text_area("Email content:")
if st.button("Classify"):
    result = predict_email(input_text)
    st.subheader("Result: " + result)

# Optional: test case
if __name__ == "__main__":
    test_email = "Congratulations! You've won a free ticket to Bahamas. Click now!"
    print("Prediction:", predict_email(test_email))
