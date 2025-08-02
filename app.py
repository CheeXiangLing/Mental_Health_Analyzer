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

# Must be first Streamlit command
st.set_page_config(page_title="Mental Health Analyzer", layout="wide")

# === Initialize NLTK Data ===
def initialize_nltk():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        with st.spinner("📥 Downloading NLTK data (first run only)..."):
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('wordnet')

initialize_nltk()

# === Load NLP Utilities ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# === Model Configuration ===
MODEL_CONFIG = {
    "BERT": {
        "folder": "bert_model",
        "zip_url": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/bert_model.zip",
        "required_files": ["config.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json", "model.safetensors"],
        "tokenizer_class": BertTokenizerFast,
        "model_class": BertForSequenceClassification
    },
    "DistilBERT": {
        "folder": "distilbert_model",
        "zip_url": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/distilbert_model.zip",
        "required_files": ["config.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json", "model.safetensors"],
        "tokenizer_class": DistilBertTokenizerFast,
        "model_class": DistilBertForSequenceClassification
    },
    "RoBERTa": {
        "folder": "roberta_model",
        "zip_url": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/roberta_model.zip",
        "required_files": ["config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "model.safetensors"],
        "tokenizer_class": RobertaTokenizerFast,
        "model_class": RobertaForSequenceClassification
    }
}

# === File Download Utility ===
def download_and_extract_zip(url, extract_to):
    try:
        # Download the zip file
        zip_path = os.path.join(extract_to, "temp.zip")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Remove the zip file
        os.remove(zip_path)
        return True
    except Exception as e:
        st.error(f"❌ Download/extract failed: {str(e)}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False

# === Model Setup ===
@st.cache_resource
def setup_model(model_name):
    config = MODEL_CONFIG[model_name]
    model_folder = config["folder"]
    os.makedirs(model_folder, exist_ok=True)
    
    # Check if all required files exist
    missing_files = [
        f for f in config["required_files"] 
        if not os.path.exists(os.path.join(model_folder, f))
    ]
    
    if missing_files:
        with st.spinner(f"📥 Downloading {model_name} model..."):
            if not download_and_extract_zip(config["zip_url"], model_folder):
                raise Exception(f"Failed to download {model_name} model")
            
        # Verify all files were extracted
        missing_files = [
            f for f in config["required_files"] 
            if not os.path.exists(os.path.join(model_folder, f))
        ]
        if missing_files:
            st.error(f"❌ Missing files after download: {', '.join(missing_files)}")
            raise FileNotFoundError(f"Missing files: {missing_files}")

# [Rest of your code remains the same...]

# === Text Processing Functions ===
def basic_clean(text):
    return text.strip()

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

# === Traditional ML Models ===
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

# === Prediction Functions ===
def predict_transformer(model_name, text):
    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = MODEL_CONFIG[model_name]
        model_folder = config["folder"]
        
        setup_model(model_name)
        
        tokenizer = config["tokenizer_class"].from_pretrained(model_folder)
        model = config["model_class"].from_pretrained(model_folder)
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
    ["BERT", "DistilBERT", "RoBERTa", "Logistic Regression", "Naive Bayes"],
    index=0
)

# Text input
text_input = st.text_area("Enter your text here:", height=150)

if st.button("Analyze", type="primary"):
    if not text_input.strip():
        st.warning("⚠️ Please enter valid text.")
    else:
        with st.spinner("🔍 Analyzing..."):
            start_time = time.time()
            
            try:
                if model_choice in ["BERT", "DistilBERT", "RoBERTa"]:
                    cleaned = basic_clean(text_input)
                    result = predict_transformer(model_choice, cleaned)
                else:
                    cleaned = clean_text(text_input)
                    vec, log_model, nb_model = load_sklearn_models()
                    model = log_model if model_choice == "Logistic Regression" else nb_model
                    result = predict_sklearn(model_choice, vec, model, cleaned)
                
                elapsed_time = time.time() - start_time
                st.success(f"""
                ✅ **Prediction using {model_choice}:**  
                **Result:** {result}  
                **Processing time:** {elapsed_time:.2f} seconds
                """)
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")

# Sidebar information
with st.sidebar:
    st.markdown("""
    ### ℹ️ About This App
    
    This tool analyzes mental health-related text and classifies it into categories:
    
    - **Normal**
    - **Depression**
    - **Anxiety**  
    - **Suicidal**  
    - **Stress**  
    - **Bipolar**  
    - **Personality Disorder**  
    
    **Model Options:**  
    - Transformers: BERT, DistilBERT, RoBERTa  
    - Traditional: Logistic Regression, Naive Bayes  
    
    *For research/educational purposes only.*
    """)

