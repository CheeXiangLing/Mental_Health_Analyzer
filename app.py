import streamlit as st

# ✅ Must be the first Streamlit command
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

# === Initialize NLTK Data with Comprehensive Error Handling ===
@st.cache_resource
def initialize_nltk():
    required_data = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt'),
        ('corpora/wordnet', 'wordnet'),
        ('', 'omw-1.4')  # Open Multilingual WordNet
    ]
    
    for path, package in required_data:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Try to find the resource
                if path:
                    nltk.data.find(path)
                break
            except LookupError:
                if attempt == max_retries - 1:
                    st.error(f"❌ Failed to load NLTK resource: {package}")
                    st.error(f"Please ensure the app has internet access to download NLTK data")
                    return False
                try:
                    nltk.download(package, quiet=True)
                except Exception as e:
                    st.warning(f"Attempt {attempt + 1} failed for {package}: {str(e)}")
                    time.sleep(1)  # Wait before retrying
    return True

# Initialize NLTK with error display
if not initialize_nltk():
    st.error("Critical NLTK resources could not be loaded. The app cannot continue.")
    st.stop()

try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    st.error(f"Failed to initialize NLTK components: {str(e)}")
    st.stop()

# === Model URLs ===
MODEL_URLS = {
    "BERT": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/bert_model.zip",
    "DistilBERT": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/distilbert_model.zip",
    "RoBERTa": "https://github.com/CheeXiangLing/Mental_Health_Analyzer/releases/download/v1.0.0/roberta_model.zip"
}

# === Download and Extract Models ===
@st.cache_resource
def download_and_extract_model(model_name):
    try:
        zip_url = MODEL_URLS.get(model_name)
        if not zip_url:
            raise ValueError(f"No URL found for {model_name}")
            
        folder = f"{model_name.lower()}_model"
        
        if not os.path.exists(folder):
            progress = st.progress(0)
            status = st.empty()
            status.info(f"⏳ Downloading {model_name} model...")
            
            # Download the model
            zip_path = f"{folder}.zip"
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.progress(min(downloaded / total_size, 1.0))
            
            # Extract the model
            status.info(f"⏳ Extracting {model_name} model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(folder)
            os.remove(zip_path)
            
            progress.empty()
            status.success(f"✅ {model_name} model ready!")
            return True
        return False
    except Exception as e:
        st.error(f"Failed to download {model_name} model: {str(e)}")
        return False

# === Text Processing Functions ===
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
        
        words = word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    except Exception as e:
        st.error(f"Text cleaning error: {str(e)}")
        return text  # Return original text if cleaning fails

# === Model Loading ===
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

# === Main App UI ===
st.title("🧠 Mental Health Sentiment Analysis")

# Model selection
col1, col2 = st.columns([1, 3])
with col1:
    model_choice = st.selectbox(
        "Choose Model:",
        ["BERT", "DistilBERT", "RoBERTa", "Logistic Regression", "Naive Bayes"],
        key="model_selector"
    )

with col2:
    text_input = st.text_area("Enter your text here:", height=150, key="text_input")

if st.button("Analyze", type="primary", key="analyze_btn"):
    if not text_input.strip():
        st.warning("⚠️ Please enter valid text.")
    else:
        with st.spinner("🔍 Analyzing input..."):
            start_time = time.time()
            
            try:
                if model_choice in ["BERT", "DistilBERT", "RoBERTa"]:
                    if not download_and_extract_model(model_choice):
                        st.error("Model download failed. Please try again.")
                        st.stop()
                    
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

# === Sidebar ===
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
    
    st.markdown("---")
    st.markdown("### 🔧 Technical Details")
    try:
        from transformers import __version__ as transformers_version
        st.code(f"""
        Python: {'.'.join(map(str, sys.version_info[:3]))}
        PyTorch: {torch.__version__}
        Transformers: {transformers_version}
        NLTK: {nltk.__version__}
        """)
    except:
        st.code("""
        Python: 3.9
        PyTorch: Installed
        Transformers: Installed
        NLTK: Installed
        """)
    
    if st.button("Clear Cache", key="clear_cache"):
        st.cache_resource.clear()
        st.success("Cache cleared!")
        st.experimental_rerun()
