# model evaluation utilities and figure saving

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def evaluate_model(model, X_test, y_test, model_name="model"):

    # make predictions
    predictions = model.predict(X_test)

    # compute accuracy
    acc = accuracy_score(y_test, predictions)

    # classification metrics
    report = classification_report(y_test, predictions)

    # confusion matrix
    cm = confusion_matrix(y_test, predictions)

    # ensure figure directory exists
    os.makedirs("/Users/bob/Documents/GitHub/NLP_LLM_RAG/NLP_projects/results/figures", exist_ok=True)

    # plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # save figure
    plt.savefig(f"/Users/bob/Documents/GitHub/NLP_LLM_RAG/NLP_projects/results/figures/{model_name}_confusion_matrix.png")

    plt.close()

    return acc, report, cm
