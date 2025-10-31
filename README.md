# Project 03: Fine-Tuning Transformer Architectures

This project explores the application of three distinct Transformer architectures (Encoder-only, Decoder-only, and Encoder-Decoder) to solve real-world NLP tasks. Models were fine-tuned using PyTorch and the Hugging Face library.

Project Tasks & Results

This project was divided into three distinct tasks, each leveraging a different model architecture.

### Task 1: Encoder-Only (BERT) — Customer Feedback Classification
* **Model:** `bert-base-uncased`
* **Dataset:** Customer Feedback Dataset
* **Objective:** Classify customer feedback as `positive`, `negative`, or `neutral`.
* **Results (on 20% test split):**
    * **Accuracy:** 1.0 (100%)
    * **F1-Score (Weighted):** 1.0 (100%)
    * *Note: A 100% score suggests the dataset may be small or highly simplified.*

### Task 2: Decoder-Only (GPT-2) — Pseudo-code to Code Generation
* **Model:** `gpt2`
* **Dataset:** SPoC (Search-based Pseudocode to Code)
* **Objective:** Generate executable C++ code from natural language pseudo-code.
* **Results (on 100 validation samples):**
    * **BLEU:** **[YOUR_BLEU_SCORE]**
    * *Note: A critical finding was that `tokenizer.padding_side = 'left'` is required for this model. Failure to set this results in a BLEU score of 0.0, as the model generates empty strings.*

### Task 3: Encoder–Decoder (T5) — Text Summarization
* **Model:** `t5-small`
* **Dataset:** CNN/DailyMail
* **Objective:** Generate concise, abstractive summaries of long news articles.
* **Results (on 21-batch validation subset):**
    * **ROUGE-1:** 0.4087 (40.9%)
    * **ROUGE-2:** 0.1940 (19.4%)
    * **ROUGE-L:** 0.2904 (29.0%)

---

## 🔧 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the demo app:**
    ```bash
    streamlit run app.py 
    ```
    *(Or `gradio app.py` if you use Gradio)*
