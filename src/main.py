import pandas as pd
import optuna

from sklearn.model_selection import train_test_split
from src.utils import preprocess_data
from src.finetune import finetune_model
from datasets import Dataset
from src.baseline import random_class_baseline, majority_class_baseline, logistic_regression_baseline
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)


# Train/eval/test split - first 70/30%, then the 30% gets split in 15% and 15%
df = pd.read_csv('../data/raw/question_classification_dataset.csv', sep=';')
train_df, rest_df = train_test_split(df, train_size=0.7)
eval_df, test_df = train_test_split(rest_df, train_size=0.5)


def objective(trial):
    # Suggest hyperparameters to optimize
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-4)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 5)


    # Preprocess data
    train_preprocessed = preprocess_data(train_df['Question'].tolist(), train_df['Category'].tolist())
    eval_preprocessed = preprocess_data(eval_df['Question'].tolist(), eval_df['Category'].tolist())

    # Convert to dataset objects
    train_dataset = Dataset.from_dict(train_preprocessed)
    eval_dataset = Dataset.from_dict(eval_preprocessed)

    # Fine-tune the model and get evaluation accuracy
    accuracy = finetune_model(train_dataset, eval_dataset, learning_rate, batch_size, num_train_epochs)
    return accuracy  # Objective to maximize

def main():

    # Create Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)

    train_questions = train_df['Question'].tolist()
    train_labels = train_df['Category'].tolist()
    test_questions = test_df['Question'].tolist()
    test_labels = test_df['Category'].tolist()

    majority_class_baseline(train_labels, test_labels)
    random_class_baseline(train_labels, test_labels)
    logistic_regression_baseline(train_questions, train_labels, test_questions, test_labels)

if __name__ == "__main__":
    main()

