import streamlit as st
import torch
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM
)
from peft import PeftModel, PeftConfig

# Page Configuration
st.set_page_config(page_title="AI Multi-Task Hub", page_icon="🤖", layout="wide")

st.title("🤖 Athar's AI Model Hub")
st.markdown("""
Welcome! This application showcases three different AI models trained/finetuned by **Athar800**.
Select a task from the sidebar to begin.
""")

# Sidebar
task = st.sidebar.selectbox(
    "Choose a Model",
    ["Sentiment Analysis (BERT)", "Code Generation (SPoC LoRA)", "Text Summarization (T5)"]
)

# ==========================================
# 🧠 MODEL LOADING FUNCTIONS (Cached)
# ==========================================

@st.cache_resource
def load_sentiment_model():
    model_id = "Athar800/bert-sentiment-custom"
    tokenizer = BertTokenizer.from_pretrained(model_id)
    model = BertForSequenceClassification.from_pretrained(model_id)
    return tokenizer, model

@st.cache_resource
def load_code_model():
    # Base GPT-2
    base_model_id = "gpt2"
    # Your LoRA Adapters
    lora_model_id = "Athar800/spoc-gpt2-lora-finetuned"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    
    # Load LoRA Adapters
    model = PeftModel.from_pretrained(base_model, lora_model_id)
    return tokenizer, model

@st.cache_resource
def load_summary_model():
    model_id = "Athar800/t5-news-summary"
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)
    return tokenizer, model

# ==========================================
# 1️⃣ TASK 1: SENTIMENT ANALYSIS
# ==========================================
if task == "Sentiment Analysis (BERT)":
    st.header("😊😐☹️ Customer Feedback Sentiment")
    st.write("Enter a customer review to detect if it's Positive, Neutral, or Negative.")
    
    user_input = st.text_area("Enter text:", "The product broke after only one use and customer service was unhelpful.")
    
    if st.button("Analyze Sentiment"):
        with st.spinner("Loading BERT model..."):
            tokenizer, model = load_sentiment_model()
            
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_idx = torch.argmax(probs).item()
            
        labels = {0: 'Negative ☹️', 1: 'Neutral 😐', 2: 'Positive 😊'}
        confidence = probs[0][pred_idx].item() * 100
        
        st.success(f"Prediction: **{labels[pred_idx]}**")
        st.info(f"Confidence: {confidence:.2f}%")

# ==========================================
# 2️⃣ TASK 2: CODE GENERATION (SPoC)
# ==========================================
elif task == "Code Generation (SPoC LoRA)":
    st.header("💻 Pseudocode to C++ Converter")
    st.write("Enter a single line of pseudocode to generate C++ code.")
    
    pseudo_input = st.text_input("Pseudocode:", "increment x1")
    
    if st.button("Generate Code"):
        with st.spinner("Loading GPT-2 + LoRA..."):
            tokenizer, model = load_code_model()
            
        prompt = f"{pseudo_input} {tokenizer.eos_token}"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=20, 
                pad_token_id=tokenizer.eos_token_id
            )
            
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part (remove input)
        generated_code = decoded.replace(pseudo_input, "").strip()
        
        st.code(generated_code, language="cpp")

# ==========================================
# 3️⃣ TASK 3: SUMMARIZATION
# ==========================================
elif task == "Text Summarization (T5)":
    st.header("📰 News Article Summarizer")
    st.write("Paste a long news article below to get a concise summary.")
    
    article_text = st.text_area("Article Text:", height=200)
    
    if st.button("Summarize"):
        with st.spinner("Loading T5 model..."):
            tokenizer, model = load_summary_model()
            
        input_text = "summarize: " + article_text
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"], 
                max_length=150, 
                min_length=40, 
                length_penalty=2.0, 
                num_beams=4, 
                early_stopping=True
            )
            
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("Summary:")
        st.write(summary)
