# run_sentiment_analysis.py
# Complete Sentiment Analysis pipeline

import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# -------------------------
# Project Paths
# -------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "imdb_reviews.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models")
FIGURES_PATH = os.path.join(BASE_DIR, "results", "figures")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

# add src to path
sys.path.append(os.path.dirname(__file__))

# -------------------------
# Import project modules
# -------------------------

from data_preprocessing import clean_text
from feature_engineering import create_tfidf
from train_model import train_logistic_regression
from evaluate_model import evaluate_model

# -------------------------
# Load Dataset
# -------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# -------------------------
# Clean Text
# -------------------------

print("Cleaning text...")

df["clean_review"] = df["review"].apply(clean_text)

# -------------------------
# Train/Test Split
# -------------------------

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42,
    stratify=df["sentiment"]
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

print("Training Logistic Regression model...")

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
    model_name="sentiment_model"
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
    os.path.join(MODEL_PATH, "sentiment_model.pkl")
)

joblib.dump(
    vectorizer,
    os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl")
)

print("\nModel and vectorizer saved successfully.")

print(f"\nFigures saved in: {FIGURES_PATH}")
