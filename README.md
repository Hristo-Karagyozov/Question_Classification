# Question Classification with DistilBERT

This project implements a **Question Classification** system using the **DistilBERT** transformer model. It processes a dataset of questions and their corresponding categories, prepares the data for input to the model, and performs text classification tasks.

## Project Structure

- `data/raw/`: Contains the raw dataset (`question_classification_dataset.csv`) with questions and categories separated by `;`.
- `src/`: Source code for the project, including preprocessing pipelines and main scripts.
  - `pipeline.py`: Contains the `DistilBertPreprocessingPipeline` class for data preprocessing.
  - `main.py`: Main script to load the dataset, preprocess the data, and prepare it for model training.
- `.venv/`: Python virtual environment for dependency isolation.

## Dataset

The dataset consists of two columns:
- **Question**: A question to classify (e.g., *What year was the battle at Waterloo fought?*).
- **Category**: The class label for the question (e.g., *Factual*, *Summarization*).

## Key Features

- **Preprocessing Pipeline**: Utilizes Hugging Face's `transformers` library to tokenize and preprocess text data for the DistilBERT model.
- **Categorical Labels**: Questions are classified into single-word categories for clarity.

## Requirements

Install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
