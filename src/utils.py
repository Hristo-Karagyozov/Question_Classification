from src.pipeline import DistilBertPreprocessingPipeline
from sklearn.metrics import accuracy_score, f1_score

def preprocess_data(texts, labels=None, model_name="distilbert-base-uncased", max_length=128):
    """
    Preprocess the data using the pipeline.
    Input:
        texts (list of str): List of raw input texts.
        labels (list of str, optional): label list.
        model_name (str): Name of the pretrained DistilBERT model.
        max_length (int): Maximum sequence length.
    Output:
        dict: Preprocessed inputs.
    """
    pipeline = DistilBertPreprocessingPipeline(model_name=model_name, max_length=max_length)
    return pipeline.preprocess_batch(texts, labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}