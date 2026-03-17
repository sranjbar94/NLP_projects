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


def evaluate_model(model, X_test, y_test, model_name="model", positive_label="positive"):

    # --------------------
    # Predictions
    # --------------------
    predictions = model.predict(X_test)

    # probability estimates if available
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]

    # --------------------
    # Metrics
    # --------------------
    acc = accuracy_score(y_test, predictions)

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=positive_label
    )

    report = classification_report(y_test, predictions)

    cm = confusion_matrix(y_test, predictions)

    # convert labels for ROC/AUC if needed
    y_binary = (y_test == positive_label).astype(int)

    auc = None
    if probs is not None:
        auc = roc_auc_score(y_binary, probs)

    # --------------------
    # Figure directory
    # --------------------
    fig_dir = os.path.join(
        os.getcwd(),
        "results",
        "figures"
    )
    os.makedirs(fig_dir, exist_ok=True)

    # --------------------
    # Confusion Matrix
    # --------------------
    plt.figure(figsize=(5, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(
        os.path.join(
            fig_dir,
            f"{model_name}_confusion_matrix.png"
        )
    )

    plt.close()

    # --------------------
    # ROC Curve
    # --------------------
    if probs is not None:

        fpr, tpr, _ = roc_curve(
            y_binary,
            probs
        )

        plt.figure(figsize=(5, 5))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {auc:.3f}"
        )

        plt.plot([0, 1], [0, 1], "--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")

        plt.legend()

        plt.savefig(
            os.path.join(
                fig_dir,
                f"{model_name}_roc_curve.png"
            )
        )

        plt.close()

    # --------------------
    # Precision-Recall Curve
    # --------------------
    if probs is not None:

        precision, recall, _ = precision_recall_curve(
            y_binary,
            probs
        )

        plt.figure(figsize=(6, 5))

        plt.plot(
            recall,
            precision
        )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {model_name}")

        plt.savefig(
            os.path.join(
                fig_dir,
                f"{model_name}_precision_recall.png"
            )
        )

        plt.close()

    # --------------------
    # Return results
    # --------------------
    return {
        "accuracy": acc,
        "f1_score": f1,
        "auc": auc,
        "classification_report": report,
        "confusion_matrix": cm
    }
