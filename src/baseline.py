import numpy as np

from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def majority_class_baseline(train_labels, test_labels):
    # Find the majority class in the training set
    majority_class = Counter(train_labels).most_common(1)[0][0]

    # Predict the majority class for all test samples
    predictions = [majority_class] * len(test_labels)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Majority Class Baseline Accuracy: {accuracy:.4f}")


def random_class_baseline(train_labels, test_labels):
    # Get class distribution
    classes, counts = np.unique(train_labels, return_counts=True)
    probabilities = counts / counts.sum()

    # Predict random classes based on the distribution
    predictions = np.random.choice(classes, size=len(test_labels), p=probabilities)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Random Class Baseline Accuracy: {accuracy:.4f}")


def logistic_regression_baseline(train_questions, train_labels, test_questions, test_labels):
    # Convert text to bag-of-words features
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_questions)
    X_test = vectorizer.transform(test_questions)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Logistic Regression Baseline Accuracy: {accuracy:.4f}")
    print(classification_report(test_labels, predictions))