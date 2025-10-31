import streamlit as st
import time

# --- App Configuration ---
st.set_page_config(
    page_title="Transformer NLP Project 03",
    layout="wide"
)

# --- App Interface ---
st.title("Project 03: Fine-Tuning Transformer Architectures")
st.markdown("This app demonstrates three distinct Transformer models fine-tuned for specific NLP tasks. **(Models are not yet loaded)**")

# Create tabs for each task
tab1, tab2, tab3 = st.tabs([
    "Task 1: Sentiment Classification (BERT)", 
    "Task 2: Code Generation (GPT-2)", 
    "Task 3: Text Summarization (T5)"
])

# --- Task 1: BERT ---
with tab1:
    st.header("Task 1: Customer Feedback Classification (BERT)")
    st.write("Enter a customer review (e.g., 'The product quality is poor!') to classify its sentiment.")
    
    text_in_1 = st.text_area("Customer Feedback:", key="bert_input")
    
    if st.button("Classify Sentiment"):
        with st.spinner("Classifying..."):
            time.sleep(1) # Simulate loading
            st.info("Model for Task 1 will be connected here.")
            st.success("**Sentiment: [Example: Positive]**")

# --- Task 2: GPT-2 ---
with tab2:
    st.header("Task 2: Pseudo-code to Code Generation (GPT-2)")
    st.write("Enter pseudo-code (e.g., 'read n', 'let A be vector') to generate C++ code.")
    
    text_in_2 = st.text_area("Pseudo-code:", key="gpt_input")
    
    if st.button("Generate Code"):
        with st.spinner("Generating C++..."):
            time.sleep(1) # Simulate loading
            st.info("Model for Task 2 will be connected here.")
            st.code("/* Example C++ code will appear here */\nint a, b, c;", language="cpp")

# --- Task 3: T5 ---
with tab3:
    st.header("Task 3: Text Summarization (T5)")
    st.write("Enter a long article to generate a short, abstractive summary.")
    
    text_in_3 = st.text_area("Article Text:", height=250, key="t5_input")
    
    if st.button("Summarize Text"):
        with st.spinner("Summarizing..."):
            time.sleep(1) # Simulate loading
            st.info("Model for Task 3 will be connected here.")
            st.success("**Generated Summary:**")
            st.write("This is a placeholder summary. The real T5 model will be connected soon.")

