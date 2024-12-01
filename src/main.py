import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import preprocess_data
from src.finetune import finetune_model
from datasets import Dataset

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)


def main():

    df = pd.read_csv('../data/raw/question_classification_dataset.csv', sep=';')

    # Train/eval/test split - first 70/30%, then the 30% gets split in 15% and 15%
    train_df, rest_df = train_test_split(df, train_size=0.7)
    eval_df, test_df = train_test_split(rest_df, train_size=0.5)

    train_preprocessed = preprocess_data(train_df['Question'].tolist(), train_df['Category'].tolist())
    eval_preprocessed = preprocess_data(eval_df['Question'].tolist(), eval_df['Category'].tolist())


    # Turning into dataset objects which the Huggingface Trainer class expects
    train_dataset = Dataset.from_dict(train_preprocessed)
    eval_dataset = Dataset.from_dict(eval_preprocessed)

    finetune_model(train_dataset, eval_dataset)

if __name__ == "__main__":
    main()
