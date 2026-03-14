# run_sentiment_analysis.py
# Complete Sentiment Analysis pipeline
# Uses src/ modules and saves confusion matrix to results/figures/

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# add src to path
import sys
sys.path.append(os.path.dirname(__file__))

from data_preprocessing import clean_text
from feature_engineering import create_tfidf
from train_model import train_logistic_regression
from evaluate_model import evaluate_model

# --- Paths ---
DATA_PATH = "/Users/bob/Documents/GitHub/NLP_LLM_RAG/NLP_projects/data/raw/"
MODEL_PATH = "../NLP_projects/models/saved_models/"
FIGURES_PATH = "../NLP_projects/results/figures/"

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

# --- Load Dataset ---
df = pd.read_csv(DATA_PATH + "imdb_reviews.csv")

# --- Clean Text ---
df["clean_review"] = df["review"].apply(clean_text)

# --- Split Dataset ---
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"],
    df["sentiment"],
    test_size=0.2,
    random_state=42
)

# --- Feature Engineering ---
X_train_vec, X_test_vec, vectorizer = create_tfidf(X_train, X_test, max_features=5000)

# --- Train Model ---
model = train_logistic_regression(X_train_vec, y_train)

# --- Evaluate Model ---
acc, report, cm = evaluate_model(
    model,
    X_test_vec,
    y_test,
    model_name="sentiment_model"
)

print("Accuracy:", acc)
print(report)

# --- Save Model and Vectorizer ---
joblib.dump(model, MODEL_PATH + "sentiment_model.pkl")
joblib.dump(vectorizer, MODEL_PATH + "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully.")
print(f"Confusion matrix saved to {FIGURES_PATH}sentiment_model_confusion_matrix.png")
