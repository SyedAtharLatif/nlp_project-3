import streamlit as st
import torch
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AutoTokenizer, AutoModelForCausalLM,
    T5Tokenizer, T5ForConditionalGeneration
)

# --- App Configuration ---
st.set_page_config(
    page_title="Transformer NLP Project 03",
    layout="wide"
)

# --- Model Loading (Cached) ---
# This ensures we only load the models once.
# Create folders for each saved model and put them in your repository.
@st.cache_resource
def load_classifier():
    # This path is relative to your GitHub repo root
    path = "./bert_classifier" 
    try:
        tokenizer = BertTokenizer.from_pretrained(path)
        model = BertForSequenceClassification.from_pretrained(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return tokenizer, model, device, None
    except Exception as e:
        return None, None, None, f"Error loading BERT model: {e}"

@st.cache_resource
def load_code_generator():
    # This path is relative to your GitHub repo root
    path = "./gpt2_code_generator_model" 
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        # This is critical for the app to work, assuming you saved the model correctly
        if tokenizer.padding_side != 'left':
            st.warning("GPT-2 Tokenizer 'padding_side' is not 'left'. Generation may fail.")
        return tokenizer, model, device, None
    except Exception as e:
        return None, None, None, f"Error loading GPT-2 model: {e}"

@st.cache_resource
def load_summarizer():
    # This path is relative to your GitHub repo root
    path = "./t5_summarization_model" 
    try:
        tokenizer = T5Tokenizer.from_pretrained(path)
        model = T5ForConditionalGeneration.from_pretrained(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return tokenizer, model, device, None
    except Exception as e:
        return None, None, None, f"Error loading T5 model: {e}"

# --- Load all models ---
with st.spinner("Loading all three models..."):
    bert_tokenizer, bert_model, bert_device, bert_error = load_classifier()
    gpt_tokenizer, gpt_model, gpt_device, gpt_error = load_code_generator()
    t5_tokenizer, t5_model, t5_device, t5_error = load_summarizer()

# --- App Interface ---
st.title("Project 03: Fine-Tuning Transformer Architectures")
st.markdown("This app demonstrates three distinct Transformer models fine-tuned for specific NLP tasks.")

# Create tabs for each task
tab1, tab2, tab3 = st.tabs([
    "Task 1: Sentiment Classification (BERT)", 
    "Task 2: Code Generation (GPT-2)", 
    "Task 3: Text Summarization (T5)"
])

# --- Task 1: BERT ---
with tab1:
    st.header("Task 1: Customer Feedback Classification (BERT)")
    if bert_error:
        st.error(bert_error)
    else:
        st.write("Enter a customer review (e.g., 'The product quality is poor!') to classify its sentiment.")
        label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        text_in_1 = st.text_area("Customer Feedback:", key="bert_input")
        
        if st.button("Classify Sentiment"):
            if text_in_1:
                with st.spinner("Classifying..."):
                    inputs = bert_tokenizer(text_in_1, return_tensors='pt', truncation=True, padding=True, max_length=128).to(bert_device)
                    with torch.no_grad():
                        outputs = bert_model(**inputs)
                        pred = torch.argmax(outputs.logits, dim=1).item()
                    
                    result = label_map[pred]
                    if result == "Positive":
                        st.success(f"**Sentiment: Positive**")
                    elif result == "Negative":
                        st.error(f"**Sentiment: Negative**")
                    else:
                        st.info(f"**Sentiment: Neutral**")
            else:
                st.warning("Please enter some text.")

# --- Task 2: GPT-2 ---
with tab2:
    st.header("Task 2: Pseudo-code to Code Generation (GPT-2)")
    if gpt_error:
        st.error(gpt_error)
    else:
        st.write("Enter pseudo-code (e.g., 'read n', 'let A be vector') to generate C++ code.")
        text_in_2 = st.text_area("Pseudo-code:", key="gpt_input")
        
        if st.button("Generate Code"):
            if text_in_2:
                with st.spinner("Generating C++..."):
                    prompt = f"{text_in_2} {gpt_tokenizer.eos_token}"
                    # Tokenizer MUST have padding_side='left'
                    inp = gpt_tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True).input_ids.to(gpt_device)
                    with torch.no_grad():
                        out = gpt_model.generate(
                            inp, 
                            max_length=512, 
                            num_beams=4, 
                            early_stopping=True, 
                            pad_token_id=gpt_tokenizer.eos_token_id
                        )
                    gen = gpt_tokenizer.decode(out[0][len(inp[0]):], skip_special_tokens=True)
                    st.code(gen, language="cpp")
            else:
                st.warning("Please enter some pseudo-code.")

# --- Task 3: T5 ---
with tab3:
    st.header("Task 3: Text Summarization (T5)")
    if t5_error:
        st.error(t5_error)
    else:
        st.write("Enter a long article to generate a short, abstractive summary.")
        text_in_3 = st.text_area("Article Text:", height=250, key="t5_input")
        
        if st.button("Summarize Text"):
            if text_in_3:
                with st.spinner("Summarizing..."):
                    prompt = "summarize: " + text_in_3
                    inputs = t5_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(t5_device)
                    with torch.no_grad():
                        out = t5_model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_length=150,
                            num_beams=4,
                            early_stopping=True
                        )
                    summary = t5_tokenizer.decode(out[0], skip_special_tokens=True)
                    st.success("**Generated Summary:**")
                    st.write(summary)
            else:
                st.warning("Please enter some text.")
