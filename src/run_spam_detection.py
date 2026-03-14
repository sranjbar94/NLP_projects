# run_spam_detection.py
# Spam Email Detection pipeline

import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# -------------------------
# Project Paths
# -------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "spam_emails.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models")
FIGURES_PATH = os.path.join(BASE_DIR, "results", "figures")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

# add src to path
sys.path.append(os.path.dirname(__file__))

# -------------------------
# Import modules
# -------------------------

from data_preprocessing import clean_text
from feature_engineering import create_tfidf
from train_model import train_logistic_regression
from evaluate_model import evaluate_model

# -------------------------
# Load Dataset
# -------------------------

print("Loading spam dataset...")

df = pd.read_csv(DATA_PATH)
# rename column to match pipeline
df = df.rename(columns={"spam": "label"})

# expected dataset structure:
# text | label
# label: 1 = spam, 0 = ham

print("Dataset size:", df.shape)

# -------------------------
# Clean Text
# -------------------------

print("Cleaning email text...")

df["clean_email"] = df["text"].apply(clean_text)

# -------------------------
# Train/Test Split
# -------------------------

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_email"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# -------------------------
# Feature Engineering
# -------------------------

print("Creating TF-IDF features...")

X_train_vec, X_test_vec, vectorizer = create_tfidf(
    X_train,
    X_test,
    max_features=5000
)

# -------------------------
# Train Model
# -------------------------

print("Training spam classifier...")

model = train_logistic_regression(
    X_train_vec,
    y_train
)

# -------------------------
# Evaluate Model
# -------------------------

print("Evaluating model...")

results = evaluate_model(
    model,
    X_test_vec,
    y_test,
    model_name="spam_detection_model",
    positive_label=1
)

print("\n----- Model Performance -----")

print("Accuracy:", results["accuracy"])
print("F1 Score:", results["f1_score"])
print("AUC:", results["auc"])

print("\nClassification Report:\n")
print(results["classification_report"])

# -------------------------
# Save Model
# -------------------------

print("\nSaving model and vectorizer...")

joblib.dump(
    model,
    os.path.join(MODEL_PATH, "spam_detection_model.pkl")
)

joblib.dump(
    vectorizer,
    os.path.join(MODEL_PATH, "spam_tfidf_vectorizer.pkl")
)

print("\nModel and vectorizer saved successfully.")

print(f"\nFigures saved in: {FIGURES_PATH}")
