import streamlit as st
import torch
import os
import requests
import zipfile
import nltk
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

# Must be the first Streamlit command
st.set_page_config(page_title="Mental Health Analyzer", layout="wide")

# === Initialize NLTK Data ===
def initialize_nltk():
    try:
        # Download required NLTK data
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt_tab', quiet=True)  # Added this line
    except Exception as e:
        st.error(f"❌ Error downloading NLTK data: {str(e)}")
        raise

# Initialize NLTK before anything else
initialize_nltk()

# === Load NLP Utilities ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# === URLs of model ZIPs from GitHub Releases ===
MODEL_URLS = {
    "BERT": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/bert_model.zip",
    "DistilBERT": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/distilbert_model.zip",
    "RoBERTa": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/roberta_model.zip"
}

# === Download and extract transformer models ===
@st.cache_resource
def download_and_extract_model(model_name):
    zip_url = MODEL_URLS.get(model_name)
    folder = f"{model_name.lower()}_model"

    # Check if model folder exists and contains required files
    required_files = {
        "BERT": ["config.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json", "model.safetensors"],
        "DistilBERT": ["config.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json", "model.safetensors"],
        "RoBERTa": ["config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "model.safetensors"]
    }
    
    # Check if all required files exist
    files_exist = all(os.path.exists(os.path.join(folder, f)) for f in required_files[model_name])
    
    if not files_exist:
        try:
            # Create directory if it doesn't exist
            os.makedirs(folder, exist_ok=True)
                
            zip_path = f"{folder}.zip"
            r = requests.get(zip_url, stream=True)
            r.raise_for_status()
                
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(folder)
            os.remove(zip_path)
                
            # Verify all files were extracted
            missing_files = [f for f in required_files[model_name] if not os.path.exists(os.path.join(folder, f))]
            if missing_files:
                st.error(f"❌ Missing files after extraction: {', '.join(missing_files)}")
                raise FileNotFoundError(f"Missing files: {missing_files}")
                    

# === Text Processing Functions ===
def basic_clean(text):
    return text.strip()

def clean_text(text):
    try:
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
    except Exception as e:
        st.error(f"❌ Error cleaning text: {str(e)}")
        raise

# === Load Sklearn Models ===
@st.cache_resource
def load_sklearn_models():
    try:
        vectorizer = joblib.load("traditional/tfidf_vectorizer.pkl")
        logistic_model = joblib.load("traditional/logistic_model.pkl")
        naive_model = joblib.load("traditional/naive_bayes_model.pkl")
        return vectorizer, logistic_model, naive_model
    except Exception as e:
        st.error(f"❌ Failed to load traditional models: {str(e)}")
        raise

# === Label Mapping ===
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
    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folder = f"{model_name.lower()}_model"

        # Ensure model is downloaded
        download_and_extract_model(model_name)

        if model_name == "BERT":
            tokenizer = BertTokenizerFast.from_pretrained(folder)
            model = BertForSequenceClassification.from_pretrained(folder)
        elif model_name == "DistilBERT":
            tokenizer = DistilBertTokenizerFast.from_pretrained(folder)
            model = DistilBertForSequenceClassification.from_pretrained(folder)
        elif model_name == "RoBERTa":
            tokenizer = RobertaTokenizerFast.from_pretrained(folder)
            model = RobertaForSequenceClassification.from_pretrained(folder)

        model.to(DEVICE)
        model.eval()

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        return label_map.get(prediction, "Unknown")
    except Exception as e:
        st.error(f"❌ Error in {model_name} prediction: {str(e)}")
        return "Error in prediction"

def predict_sklearn(model_type, vectorizer, model, text):
    try:
        vectorized = vectorizer.transform([text])
        prediction = model.predict(vectorized)[0]
        return label_map.get(prediction, "Unknown")
    except Exception as e:
        st.error(f"❌ Error in {model_type} prediction: {str(e)}")
        return "Error in prediction"

# === Streamlit UI ===
st.title("🧠 Mental Health Sentiment Analysis")

# Model selection
model_choice = st.selectbox(
    "Choose Model:",
    ["BERT", "DistilBERT", "RoBERTa", "Logistic Regression", "Naive Bayes"]
)

text_input = st.text_area("Enter your text here:", height=150)

if st.button("Analyze"):
    if not text_input.strip():
        st.warning("⚠️ Please enter valid text.")
    else:
        with st.spinner("🔍 Analyzing input..."):
            try:
                if model_choice in ["BERT", "DistilBERT", "RoBERTa"]:
                    cleaned = basic_clean(text_input)
                    result = predict_transformer(model_choice, cleaned)
                else:
                    cleaned = clean_text(text_input)
                    vec, log_model, nb_model = load_sklearn_models()
                    model = log_model if model_choice == "Logistic Regression" else nb_model
                    result = predict_sklearn(model_choice, vec, model, cleaned)

                st.success(f"✅ Prediction using {model_choice}: {result}")
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

# Sidebar information
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


