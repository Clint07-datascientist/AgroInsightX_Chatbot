# AgroInsightX_Chatbot
---
# AgroInsightX_Chatbot: Fine-Tuned BERT for Agricultural Question Answering

## 1. Project Overview
AgroInsightX is a specialized Question Answering (QA) chatbot developed to assist farmers and agricultural enthusiasts by providing quick and accurate information on agricultural topics. This project fine-tunes a **BERT-based model** on a custom dataset of agriculture-specific questions and answers to create a domain-focused QA system.

The chatbot functions as a **Reading Comprehension QA model**, extracting the most relevant span of text (the answer) from a provided context based on the user's question.

## 2. Dataset Details
The model is trained using the **`agriculture-qa-english-only.parquet`** dataset.

| Feature | Details |
| :--- | :--- |
| **Source File** | `data/agriculture-qa-english-only.parquet` |
| **Total Entries** | 22,615 question-answer pairs |
| **Columns** | `question`, `answers` (single string field for context/answer) |
| **Question Length (Mean)** | ~8.8 words per question |
| **Answer Length (Mean)** | ~23.4 words per answer/context |

## 3. Model Architecture and Training
The project uses the Hugging Face `transformers` library to fine-tune a pre-trained language model.

* **Base Model:** `bert-base-uncased`
* **Model Class:** `AutoModelForQuestionAnswering`
* **Training Parameters:**
    * **Total Epochs:** 3
    * **Batch Size (Train/Eval):** 16
    * **Optimizer:** Adam with 500 warmup steps
    * **Best Model Saving:** Enabled (`load_best_model_at_end=True`) based on the evaluation metric.
    * **Hub:** The model is saved to the Hugging Face Hub at `Clint07-datascientist/agri-qa-bert-fine-tuned`.

## 4. Performance Metrics
The training was configured to track standard Question Answering metrics:

| Metric | Description | Target Performance (Example Placeholder) |
| :--- | :--- | :--- |
| **F1 Score** | Harmonic mean of Precision and Recall. Measures the overlap between the predicted and true answers. | **89.5%** |
| **Exact Match (EM)** | Measures the percentage of predictions that exactly match one of the ground-truth answers. | **85.1%** |
| **Evaluation Loss** | The loss value on the validation set. | **0.25** |

*(Note: The `chatbot_training.ipynb` output showed partial training. Replace the placeholder metrics above with the final `trainer.evaluate()` results after full execution.)*

## 5. Setup and Installation

### A. Prerequisites
1.  **Python:** Python 3.8+ is recommended.
2.  **Git:** For cloning the repository.

### B. Installation Steps
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Clint07-datascientist/AgroInsightX_Chatbot.git](https://github.com/Clint07-datascientist/AgroInsightX_Chatbot.git)
    cd AgroInsightX_Chatbot
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Training/Setup:**
    Execute the cells in `chatbot_training.ipynb` to download the dataset, preprocess the data, and train the model. The trained model will be saved to the `./final_model/` directory.

## 6. How to Run the Chatbot (Inference)

After the model is trained and saved, you can use the Hugging Face `pipeline` for simple, quick inference.

**Note:** The model requires a **context** paragraph to search for the answer, as it is a span-extraction Question Answering model.

### Python Example:

```python
from transformers import pipeline

# 1. Load the fine-tuned model and tokenizer
# Replace './final_model' with your saved directory if different
qa_pipeline = pipeline(
    "question-answering",
    model="Clint07-datascientist/agri-qa-bert-fine-tuned", # Or local directory: "./final_model"
    tokenizer="bert-base-uncased"
)

# 2. Define the context and question
CONTEXT = "Crop rotation is the practice of growing a series of different types of crops in the same area across a sequence of growing seasons. This helps to prevent soil erosion and depletion. The recommended fertilizer for paddy is NPK 10-26-26."

QUESTION = "what is the recommended fertilizer"

# 3. Get the answer
result = qa_pipeline(question=QUESTION, context=CONTEXT)

print(f"Question: {QUESTION}")
print(f"Answer: {result['answer']}")
print(f"Confidence Score: {result['score']:.4f}")
