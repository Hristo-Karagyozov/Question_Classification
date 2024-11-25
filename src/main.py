import pandas as pd

from src.pipeline import DistilBertPreprocessingPipeline

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)


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


if __name__ == "__main__":
    df = pd.read_csv('../data/raw/question_classification_dataset.csv', sep=';')
    # Converting to lists; the tokenizer expects python native types
    inputs = df['Question'].tolist()
    targets = df['Category'].tolist()
    preprocessed_data = preprocess_data(inputs, targets)
