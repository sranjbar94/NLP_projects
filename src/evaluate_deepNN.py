# src/evaluate_deepNN.py
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def evaluate_model(model, data_loader, device='cpu', label_encoder=None, model_name="model"):
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Decode labels
    if label_encoder:
        y_true_dec = label_encoder.inverse_transform(y_true)
        y_pred_dec = label_encoder.inverse_transform(y_pred)
    else:
        y_true_dec, y_pred_dec = y_true, y_pred
    
    # Metrics
    report = classification_report(y_true_dec, y_pred_dec, zero_division=0)
    cm = confusion_matrix(y_true_dec, y_pred_dec)
    
    # Plot confusion matrix
    fig_dir = f"./results/figures"
    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues")
    plt.xlabel("Predict")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"{fig_dir}/{model_name}_confusion_matrix.png")
    plt.close()
    
    return report, cm
