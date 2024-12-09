import gradio as gr
from transformers import AutoModelForSequenceClassification, DistilBertTokenizer

import torch

from src.pipeline import DistilBertPreprocessingPipeline
from src.utils import preprocess_data

label_dict = {
    0:"Clarification",
    1:"Factual",
    2:"Operational",
    3:"Summarization"
}


# Load the fine-tuned model
MODEL_PATH = "../src/distillbert-finetuned-question-classifier/checkpoint-24"
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
pipeline = DistilBertPreprocessingPipeline()

# Function to process inputs and return predictions
def predict(question):
    preprocessed = preprocess_data(question)
    input_ids = preprocessed["input_ids"]
    attention_mask = preprocessed["attention_mask"]
    ids = torch.tensor(input_ids)
    mask = torch.tensor(attention_mask)

    with torch.no_grad():
        outputs = model(input_ids=ids, attention_mask=mask)
        predictions = torch.argmax(outputs.logits, dim=1).item()

    return label_dict[predictions]


# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter a Question"),
    outputs=gr.Textbox(label="Predicted Category"),
    title="Question Classification API",
    description="Enter a question, and the model will classify it into a category.",
)

# Enable shareable link
interface.launch(share=True)
