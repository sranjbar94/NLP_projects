# run_text_classification.py
# News Topic Classification pipeline with Gradient Boosting or MLP

import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "news_dataset.json")
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models")
FIGURES_PATH = os.path.join(BASE_DIR, "results", "figures")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

# Add src to path
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

# -------------------------
# Imports from src/
# -------------------------
from data_preprocessing import clean_text
from feature_engineering import create_tfidf
from evaluate_model_multiclass import evaluate_model_multiclass
from train_gradient_boosting import train_gradient_boosting
from train_mlp import train_mlp

# -------------------------
# Model Selection
# -------------------------
MODEL_TYPE = "mlp"  # "gb" for Gradient Boosting, "mlp" for Neural Network

# -------------------------
# Load Dataset
# -------------------------
print("Loading news dataset...")
df = pd.read_json(DATA_PATH, lines=True)

# Use headline as text, category as label
df = df.rename(columns={"headline": "text", "category": "label"})
print("Columns:", df.columns)
print("Dataset size:", df.shape)
print("Sample data:\n", df[["text", "label"]].head())

# -------------------------
# Clean Text
# -------------------------
print("Cleaning text...")
df["clean_text"] = df["text"].apply(clean_text)

# -------------------------
# Split Dataset
# -------------------------
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# -------------------------
# TF-IDF Feature Engineering
# -------------------------
print("Creating TF-IDF features...")
X_train_vec, X_test_vec, vectorizer = create_tfidf(
    X_train,
    X_test,
    max_features=8000
)

# -------------------------
# Train Model
# -------------------------
print(f"Training model: {MODEL_TYPE} ...")

if MODEL_TYPE == "gb":
    # Gradient Boosting requires dense input
    X_train_input = X_train_vec.toarray()
    X_test_input = X_test_vec.toarray()
    model = train_gradient_boosting(X_train_input, y_train)
elif MODEL_TYPE == "mlp":
    # MLP can handle sparse
    X_train_input = X_train_vec
    X_test_input = X_test_vec
    model = train_mlp(X_train_input, y_train)
else:
    raise ValueError("Invalid MODEL_TYPE. Choose 'gb' or 'mlp'.")

# -------------------------
# Evaluate Model
# -------------------------
print("Evaluating model...")
results = evaluate_model_multiclass(
    model,
    X_test_input,
    y_test,
    model_name=f"news_topic_classifier_{MODEL_TYPE}",
    fig_dir=FIGURES_PATH
)

# Print key metrics
print("\nEvaluation Results:")
print("Accuracy:", results["accuracy"])
print("Weighted F1-score:", results["weighted_f1"])
print("Confusion matrix saved to:", os.path.join(FIGURES_PATH, f"news_topic_classifier_{MODEL_TYPE}_confusion_matrix.png"))

# -------------------------
# Save Model and Vectorizer
# -------------------------
print("Saving model and vectorizer...")
joblib.dump(model, os.path.join(MODEL_PATH, f"news_topic_classifier_{MODEL_TYPE}.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_PATH, f"news_tfidf_vectorizer_{MODEL_TYPE}.pkl"))

print(f"Model and vectorizer saved successfully in {MODEL_PATH}")
print(f"Figures saved in {FIGURES_PATH}")
