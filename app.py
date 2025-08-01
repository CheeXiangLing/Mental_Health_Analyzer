import streamlit as st
import torch
from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    RobertaTokenizerFast, RobertaForSequenceClassification
)
import joblib
import time
import nltk
from nltk.tokenize import word_tokenize
import emoji
import html

# Download required NLTK data
nltk.download('punkt')

# === Basic cleaning ===
def basic_clean(text):
    text = text.strip()
    text = emoji.demojize(text)
    text = html.unescape(text)
    tokens = word_tokenize(text)
    return ' '.join(tokens)

# === Load Traditional ML Models ===
def load_sklearn_models():
    vectorizer = joblib.load("traditional/tfidf_vectorizer.pkl")
    log_model = joblib.load("traditional/log_model.pkl")
    nb_model = joblib.load("traditional/nb_model.pkl")
    return vectorizer, log_model, nb_model

# === Load Transformer Models ===
def load_transformer_model(model_name):
    if model_name == "BERT":
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("transformers/bert")
    elif model_name == "DistilBERT":
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("transformers/distilbert")
    elif model_name == "RoBERTa":
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained("transformers/roberta")
    else:
        raise ValueError("Unknown transformer model")
    return tokenizer, model

# === Predict with Traditional Model ===
def predict_traditional(text, vectorizer, model):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return prediction

# === Predict with Transformer Model ===
def predict_transformer(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return torch.argmax(probs).item()

# === Streamlit UI ===
st.title("Mental Health Sentiment Analyzer")
st.write("Enter a sentence related to mental health:")

text_input = st.text_area("Text Input", height=150)
model_type = st.selectbox("Select model type:", ["Traditional", "Transformer"])

if model_type == "Traditional":
    model_choice = st.selectbox("Choose Traditional Model:", ["Logistic Regression", "Naive Bayes"])
else:
    model_choice = st.selectbox("Choose Transformer Model:", ["BERT", "DistilBERT", "RoBERTa"])

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = basic_clean(text_input)
        with st.spinner("Analyzing..."):
            time.sleep(1)
            if model_type == "Traditional":
                vectorizer, log_model, nb_model = load_sklearn_models()
                model = log_model if model_choice == "Logistic Regression" else nb_model
                prediction = predict_traditional(cleaned, vectorizer, model)
            else:
                tokenizer, model = load_transformer_model(model_choice)
                prediction = predict_transformer(cleaned, tokenizer, model)

        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.success(f"Prediction: {label_map[prediction]}")
