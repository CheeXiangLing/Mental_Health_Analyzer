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

# === Download NLTK data if not present ===
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    with st.spinner("📥 Downloading NLTK data (first run only)..."):
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        st.success("✅ NLTK data downloaded successfully!")

# === Load stopwords and lemmatizer ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# === URLs of tokenizer ZIPs from GitHub Releases ===
TOKENIZER_URLS = {
    "BERT": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/ini/bert_model.zip",
    "DistilBERT": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/ini/distilbert_model.zip",
    "RoBERTa": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/ini/roberta_model.zip"
}

# === Download and extract tokenizer files if not already present ===
@st.cache_resource
def download_tokenizer_files(model_name):
    model_folder = f"{model_name.lower()}_model"
    zip_url = TOKENIZER_URLS.get(model_name)
    
    # Check if tokenizer files already exist
    required_files = {
        "BERT": ["config.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"],
        "DistilBERT": ["config.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"],
        "RoBERTa": ["config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]
    }
    
    # Check if all required files exist
    files_exist = all(os.path.exists(os.path.join(model_folder, f)) for f in required_files[model_name])
    
    if not files_exist:
        st.info(f"📦 Downloading {model_name} tokenizer files...")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_folder, exist_ok=True)
        
        try:
            # Download the zip file
            zip_path = f"{model_folder}_tokenizer.zip"
            r = requests.get(zip_url, stream=True)
            r.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            st.info(f"📂 Extracting {model_name} tokenizer files...")
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_folder)
            
            # Remove the zip file after extraction
            os.remove(zip_path)
            
            st.success(f"✅ {model_name} tokenizer files downloaded and extracted successfully!")
            
        except Exception as e:
            st.error(f"❌ Error downloading {model_name} tokenizer files: {str(e)}")
            raise e

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
    folder = f"{model_name.lower()}_model"

    # Ensure tokenizer files are downloaded
    download_tokenizer_files(model_name)

    try:
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
        st.error(f"Error loading {model_name} model: {str(e)}")
        return "Error in prediction"

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

# === Sidebar info ===
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

