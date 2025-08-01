import streamlit as st
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
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

initialize_nltk()

# === Load stopwords and lemmatizer ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# === Corrected Model URLs ===
MODEL_URLS = {
    "BERT": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/bert_model.zip",
    "DistilBERT": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/distilbert_model.zip",
    "RoBERTa": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/roberta_model.zip"
}

# === Improved Download Function ===
@st.cache_resource
def download_and_extract_model(model_name):
    try:
        zip_url = MODEL_URLS.get(model_name)
        if not zip_url:
            raise ValueError(f"No URL found for {model_name}")
            
        folder = f"{model_name.lower()}_model"
        
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            zip_path = os.path.join(folder, f"{model_name}.zip")
            
            # Show download progress
            progress_text = st.empty()
            progress_text.info(f"📦 Downloading {model_name} model...")
            progress_bar = st.progress(0)
            
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = min(downloaded / total_size, 1.0)
                        progress_bar.progress(progress)
            
            # Show extraction progress
            progress_text.info(f"📂 Extracting {model_name} model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(folder)
            
            os.remove(zip_path)
            progress_text.empty()
            progress_bar.empty()
            
            return True
        return False
    except Exception as e:
        st.error(f"❌ Failed to download {model_name}: {str(e)}")
        return False

# === Text Cleaning Functions ===
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
        st.error(f"Prediction error: {str(e)}")
        return "Error"

def predict_sklearn(model_type, vectorizer, model, text):
    try:
        vectorized = vectorizer.transform([text])
        prediction = model.predict(vectorized)[0]
        return label_map.get(prediction, "Unknown")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error"

# === Streamlit UI ===
st.set_page_config(page_title="Mental Health Analyzer", page_icon="🧠", layout="wide")

st.title("🧠 Mental Health Sentiment Analysis")

# Model selection
col1, col2 = st.columns([1, 3])
with col1:
    model_choice = st.selectbox(
        "Choose Model:",
        ["BERT", "DistilBERT", "RoBERTa", "Logistic Regression", "Naive Bayes"]
    )

with col2:
    text_input = st.text_area("Enter your text here:", height=150)

if st.button("Analyze", type="primary"):
    if not text_input.strip():
        st.warning("⚠️ Please enter valid text.")
    else:
        with st.spinner("🔍 Analyzing input..."):
            start_time = time.time()
            
            if model_choice in ["BERT", "DistilBERT", "RoBERTa"]:
                # Ensure model is downloaded first
                if not download_and_extract_model(model_choice):
                    st.error("Failed to download model. Please try again.")
                    st.stop()
                
                cleaned = basic_clean(text_input)
                result = predict_transformer(model_choice, cleaned)
            else:
                cleaned = clean_text(text_input)
                vec, log_model, nb_model = load_sklearn_models()
                if vec is None or log_model is None or nb_model is None:
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

# === Sidebar ===
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app analyzes mental health-related text and classifies it into psychological conditions.
    It supports both transformer models (BERT, DistilBERT, RoBERTa) and traditional ML models.
    
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
    
    st.markdown("---")
    st.markdown("### 🔧 Technical Details")
    try:
        from transformers import __version__ as transformers_version
        st.code(f"""
        Python: 3.9.16
        PyTorch: {torch.__version__}
        Transformers: {transformers_version}
        """)
    except:
        st.code("""
        Python: 3.9.16
        PyTorch: Installed
        Transformers: Installed
        """)
    
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared!")
