# evaluate_model_multiclass.py
# New evaluation module for multi-class classification with visualizations

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize

def evaluate_model_multiclass(model, X_test, y_test, model_name="model", fig_dir=None):
    """
    Evaluate a multi-class or binary classifier and save performance figures.

    Parameters:
    - model: trained classifier
    - X_test: features
    - y_test: true labels
    - model_name: string, used for figure filenames
    - fig_dir: directory to save figures
    """

    if fig_dir is None:
        fig_dir = os.path.join(os.getcwd(), "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # --- Predictions ---
    predictions = model.predict(X_test)

    # --- Metrics ---
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")
    report = classification_report(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=np.unique(y_test))

    print("\nAccuracy:", acc)
    print("Weighted F1 Score:", f1)
    print("\nClassification Report:\n", report)

    # --- Confusion Matrix ---
    plt.figure(figsize=(8,6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test)
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_file = os.path.join(fig_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_file)
    plt.close()

    # --- Precision-Recall per class (multi-class) ---
    try:
        if hasattr(model, "predict_proba"):
            classes = np.unique(y_test)
            y_test_bin = label_binarize(y_test, classes=classes)
            probs = model.predict_proba(X_test)

            plt.figure(figsize=(8,6))
            for i, class_label in enumerate(classes):
                precision, recall, _ = precision_recall_curve(y_test_bin[:,i], probs[:,i])
                plt.plot(recall, precision, label=class_label)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve per Class - {model_name}")
            plt.legend()
            plt.tight_layout()
            pr_file = os.path.join(fig_dir, f"{model_name}_precision_recall.png")
            plt.savefig(pr_file)
            plt.close()
    except Exception as e:
        print("Precision-Recall plot skipped:", e)

    # --- Optional: Predicted vs True label distribution ---
    plt.figure(figsize=(8,5))
    import pandas as pd
    df_compare = pd.DataFrame({"true": y_test, "pred": predictions})
    df_compare.groupby("true")["pred"].value_counts().unstack().plot(
        kind="bar", stacked=True, figsize=(10,6)
    )
    plt.title(f"Predicted vs True Labels - {model_name}")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    dist_file = os.path.join(fig_dir, f"{model_name}_pred_vs_true.png")
    plt.savefig(dist_file)
    plt.close()

    print(f"\nFigures saved to: {fig_dir}")

    return {
        "accuracy": acc,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": cm
    }