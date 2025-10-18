# AgroInsightX_Chatbot
---
# AgroInsightX_Chatbot: Fine-Tuned BERT for Agricultural Question Answering
--
## [Demo Video](https://youtu.be/BsqZBagZlQI)

## 1. Project Overview
AgroInsightX is a specialized Question Answering (QA) chatbot developed to assist farmers and agricultural enthusiasts by providing quick and accurate information on agricultural topics. This project fine-tunes a **BERT-based model** on a custom dataset of agriculture-specific questions and answers to create a domain-focused QA system.

The chatbot functions as a **Reading Comprehension QA model**, extracting the most relevant span of text (the answer) from a provided context based on the user's question.

## 2. Dataset Details
The model is trained on the **Agriculture QA** dataset, sourced from a local Parquet file.

| Feature | Details |
| :--- | :--- |
| **Source Model Checkpoint** | `bert-base-uncased` |
| **Total Training Pairs** | 22,615 question-answer pairs |
| **Data Fields** | `question`, `answers` (used as context/source for answer extraction) |
| **Data Split** | 90% Training, 10% Validation (2,262 examples) |

![Image](https://github.com/user-attachments/assets/5001bba9-c9f6-487c-b7a3-cf8c28317b5f)

![Image](https://github.com/user-attachments/assets/31dc79c2-e458-4340-a3e9-72b14e8e192b)

![Image](https://github.com/user-attachments/assets/66d8ee5c-9f92-40d5-ae40-f0533d6453bf)

![Image](https://github.com/user-attachments/assets/91a97f93-ab9f-4ff0-88f6-ea247a29015e)

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

| Parameter | Value |
| :--- | :--- |
| **Total Epochs** | 3 |
| **Training Batch Size** | 16 (per device) |
| **Hardware** | Google Colab T4 GPU |
| **Model Saving** | Best model saved based on lowest validation loss. |

## 4. Performance Metrics (Final Results)

The model was evaluated on the validation set (2,262 examples) after training completion.

### Training Loss Progression
| Epoch | Training Loss (Average) | Validation Loss |
| :--- | :--- | :--- |
| 1 | 0.072800 | 0.008774 |
| 2 | 0.004200 | 0.005134 |
| **3 (Final)** | 0.006300 | **0.004831** |

### Final Question Answering Metrics
The definitive performance metrics confirm the model's high accuracy in extracting answers:

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Exact Match (EM)** | **84.87%** | Percentage of predictions that match the ground truth **exactly**. |
| **F1 Score** | **89.26%** | Harmonic mean of precision and recall (token overlap). |
| **Final Eval Loss** | **0.004996** | The loss on the validation set during the final evaluation run. |

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
