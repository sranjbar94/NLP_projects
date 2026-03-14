# model evaluation utilities and figure saving

import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)

def evaluate_model(model, X_test, y_test, model_name="model"):

    # make predictions
    predictions = model.predict(X_test)

    # probabilities for AUC / ROC
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:,1]
    else:
        probs = None

    # metrics
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="binary", pos_label="positive")

    report = classification_report(y_test, predictions)

    cm = confusion_matrix(y_test, predictions)

    if probs is not None:
        auc = roc_auc_score(y_test, probs)
    else:
        auc = None

    # ensure figure directory exists
    fig_dir = "/Users/bob/Documents/GitHub/NLP_LLM_RAG/NLP_projects/results/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # --------------------
    # Confusion Matrix
    # --------------------
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(f"{fig_dir}/{model_name}_confusion_matrix.png")
    plt.close()

    # --------------------
    # ROC Curve
    # --------------------
    if probs is not None:

        fpr, tpr, _ = roc_curve(y_test, probs)

        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0,1],[0,1],'--')

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend()

        plt.savefig(f"{fig_dir}/{model_name}_roc_curve.png")
        plt.close()

    # --------------------
    # Precision Recall Curve
    # --------------------
    if probs is not None:

        precision, recall, _ = precision_recall_curve(y_test, probs)

        plt.figure(figsize=(6,5))
        plt.plot(recall, precision)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {model_name}")

        plt.savefig(f"{fig_dir}/{model_name}_precision_recall.png")
        plt.close()

    return {
        "accuracy": acc,
        "f1_score": f1,
        "auc": auc,
        "classification_report": report,
        "confusion_matrix": cm
    }
