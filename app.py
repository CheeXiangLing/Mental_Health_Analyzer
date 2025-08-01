import streamlit as st
import torch
from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    RobertaTokenizerFast, RobertaForSequenceClassification
)
import joblib
import time
import html
import emoji
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# === Load stopwords and lemmatizer ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# === Basic clean for transformers ===
def basic_clean(text):
    return text.strip()

# === Full clean for traditional models ===
def clean_text(text):
    text = html.unescape(text)
    text = emoji.replace_emoji(text, replace='')

    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

# === Load Sklearn Models ===
def load_sklearn_models():
    vectorizer = joblib.load("traditional/tfidf_vectorizer.pkl")
    logistic_model = joblib.load("traditional/logistic_model.pkl")
    naive_model = joblib.load("traditional/naive_bayes_model.pkl")
    return vectorizer, logistic_model, naive_model

# === Label Map ===
label_map = {
    0: "Normal",
    1: "Depression",
    2: "Anxiety",
    3: "Suicidal",
    4: "Stress",
    5: "Bipolar",
    6: "Personality disorder"
}

# === Inference Functions ===
def predict_transformer(model_name, text):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "BERT":
        model_dir = "bert_model"
        tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        model = BertForSequenceClassification.from_pretrained(model_dir)
    elif model_name == "DistilBERT":
        model_dir = "distilbert_model"
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    elif model_name == "RoBERTa":
        model_dir = "roberta_model"
        tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
        model = RobertaForSequenceClassification.from_pretrained(model_dir)

    model.to(DEVICE)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return label_map.get(prediction, "Unknown")

def predict_sklearn(model_type, vectorizer, model, text):
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    return label_map.get(prediction, "Unknown")

# === Streamlit UI ===
st.title("🧠 Mental Health Sentiment Analysis")

# Select model
model_choice = st.selectbox(
    "Choose Model:",
    ["BERT", "DistilBERT", "RoBERTa", "Logistic Regression", "Naive Bayes"]
)

text_input = st.text_area("Enter your text here:")

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter valid text.")
    else:
        with st.spinner("🔍 Analyzing input..."):
            time.sleep(1)

            if model_choice in ["BERT", "DistilBERT", "RoBERTa"]:
                cleaned = basic_clean(text_input)
                result = predict_transformer(model_choice, cleaned)
            else:
                cleaned = clean_text(text_input)
                vec, log_model, nb_model = load_sklearn_models()
                model = log_model if model_choice == "Logistic Regression" else nb_model
                result = predict_sklearn(model_choice, vec, model, cleaned)

        st.success(f"✅ Prediction using {model_choice}: {result}")

## Add a about section
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown(
        '<u><b>Mental Health Sentiment Analyzer</b></u><br><br>'
        'This app analyzes mental health-related text and classifies it into one of several psychological conditions. '
        'It supports both modern transformer-based models (<b>BERT</b>, <b>DistilBERT</b>, <b>RoBERTa</b>) and traditional machine learning models '
        '(<b>Logistic Regression</b>, <b>Naive Bayes</b>).<br><br>'
        'Among the transformer models, <b>BERT</b> delivers the best performance. For traditional models, <b>Logistic Regression</b> performs the best overall.<br><br>'
        '<b>Detected Categories:</b><br>'
        '- Normal<br>'
        '- Depression<br>'
        '- Anxiety<br>'
        '- Suicidal<br>'
        '- Stress<br>'
        '- Bipolar<br>'
        '- Personality Disorder<br><br>'
        '<i>This tool is intended for research and educational purposes only.</i>',
        unsafe_allow_html=True
    )



