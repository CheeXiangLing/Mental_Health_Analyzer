import streamlit as st

# ✅ Must be set before any other Streamlit elements
st.set_page_config(page_title="Mental Health Analyzer", page_icon="🧠", layout="wide")

import torch
import os
import requests
import zipfile
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
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# === Initialize NLTK Data ===
@st.cache_resource
def initialize_nltk():
    try:
        # Download required NLTK data
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')  # Required for lemmatization
        
        # Verify downloads
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
    except Exception as e:
        st.error(f"Failed to initialize NLTK: {str(e)}")
        raise

try:
    initialize_nltk()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    st.error(f"Critical error: {str(e)}")
    st.stop()

# === URLs of model ZIPs from GitHub Releases ===
MODEL_URLS = {
    "BERT": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/bert_model.zip",
    "DistilBERT": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/distilbert_model.zip",
    "RoBERTa": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/roberta_model.zip"
}

# === Download and extract transformer models if not already present ===
@st.cache_resource
def download_and_extract_model(model_name):
    try:
        zip_url = MODEL_URLS.get(model_name)
        if not zip_url:
            raise ValueError(f"No URL found for {model_name}")
            
        folder = f"{model_name.lower()}_model"
        
        if not os.path.exists(folder):
            with st.spinner(f"📦 Downloading {model_name} model..."):
                zip_path = f"{folder}.zip"
                r = requests.get(zip_url, stream=True)
                r.raise_for_status()
                
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                with st.spinner(f"📂 Extracting {model_name} model..."):
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(folder)
                    os.remove(zip_path)
            
            return True
        return False
    except Exception as e:
        st.error(f"Failed to download {model_name} model: {str(e)}")
        return False

# === Text Cleaning Functions ===
def basic_clean(text):
    """Simple cleaning for transformer models"""
    return text.strip()

def clean_text(text):
    """Advanced cleaning for traditional models"""
    try:
        text = html.unescape(text)
        text = emoji.replace_emoji(text, replace='')
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        
        # Tokenize with error handling
        try:
            words = word_tokenize(text)
            words = [word for word in words if word not in stop_words]
            words = [lemmatizer.lemmatize(word) for word in words]
            return ' '.join(words)
        except Exception as e:
            st.error(f"Text processing error: {str(e)}")
            return text  # Return partially cleaned text
        
    except Exception as e:
        st.error(f"Text cleaning failed: {str(e)}")
        return text  # Return original text if cleaning fails

# === Load Sklearn Models ===
@st.cache_resource
def load_sklearn_models():
    try:
        vectorizer = joblib.load("traditional/tfidf_vectorizer.pkl")
        logistic_model = joblib.load("traditional/logistic_model.pkl")
        naive_model = joblib.load("traditional/naive_bayes_model.pkl")
        return vectorizer, logistic_model, naive_model
    except Exception as e:
        st.error(f"Failed to load traditional models: {str(e)}")
        return None, None, None

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

# === Prediction Functions ===
def predict_transformer(model_name, text):
    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        folder = f"{model_name.lower()}_model"

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
        st.error(f"Transformer prediction error: {str(e)}")
        return "Error"

def predict_sklearn(model_type, vectorizer, model, text):
    try:
        vectorized = vectorizer.transform([text])
        prediction = model.predict(vectorized)[0]
        return label_map.get(prediction, "Unknown")
    except Exception as e:
        st.error(f"Traditional model prediction error: {str(e)}")
        return "Error"

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
            start_time = time.time()
            
            try:
                if model_choice in ["BERT", "DistilBERT", "RoBERTa"]:
                    cleaned = basic_clean(text_input)
                    result = predict_transformer(model_choice, cleaned)
                else:
                    cleaned = clean_text(text_input)
                    vec, log_model, nb_model = load_sklearn_models()
                    if None in [vec, log_model, nb_model]:
                        st.error("Traditional models failed to load")
                        st.stop()
                    
                    model = log_model if model_choice == "Logistic Regression" else nb_model
                    result = predict_sklearn(model_choice, vec, model, cleaned)
                
                elapsed = time.time() - start_time
                st.success(f"""
                **✅ Analysis Results**  
                **Model:** {model_choice}  
                **Prediction:** {result}  
                **Processing Time:** {elapsed:.2f} seconds
                """)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# === Sidebar info ===
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app analyzes mental health-related text and classifies it into psychological conditions.
    
    **Model Recommendations:**
    - Best accuracy: BERT
    - Fastest: DistilBERT
    - Lightweight: Logistic Regression
    
    **Detected Conditions:**
    - Normal
    - Depression
    - Anxiety
    - Suicidal
    - Stress
    - Bipolar
    - Personality Disorder
    
    *For research/educational purposes only.*
    """)
    
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared!")
