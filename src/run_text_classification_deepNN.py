# src/run_text_classification_deepNN.py
import os
import pandas as pd
import torch
import joblib

from data_loader import create_dataloaders
from train_deepNN import TextClassifier, train_model
from evaluate_deepNN import evaluate_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 64
MAX_LEN = 128
EPOCHS = 5
EMBED_DIM = 100

# Paths
DATA_PATH = "./data/raw/news_dataset.json"
MODEL_PATH = "./models/saved_models/"

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs("./results/figures", exist_ok=True)

# --- Load Dataset ---
print("Loading news dataset...")
df = pd.read_json(DATA_PATH, lines=True)

# Combine headline + short description
df['text'] = df['headline'] + " " + df['short_description']
df['label'] = df['category']

print("Dataset size:", df.shape)
print("Sample data:\n", df[['text','label']].head())

# --- Create DataLoaders ---
train_loader, val_loader, label_encoder = create_dataloaders(df, text_column='text', label_column='label',
                                                            batch_size=BATCH_SIZE, max_len=MAX_LEN)

# --- Initialize model ---
num_classes = len(label_encoder.classes_)
vocab_size = 30522  # if using DistilBERT tokenizer, else adjust
model = TextClassifier(vocab_size=vocab_size, embed_dim=EMBED_DIM, num_classes=num_classes)

# --- Train model ---
print("Training model...")
model = train_model(model, train_loader, val_loader, device=DEVICE, epochs=EPOCHS)

# --- Evaluate model ---
print("Evaluating model...")
report, cm = evaluate_model(model, val_loader, device=DEVICE, label_encoder=label_encoder, model_name="news_deepNN")
print(report)

# --- Save model ---
torch.save(model.state_dict(), os.path.join(MODEL_PATH, "news_deepNN.pth"))
joblib.dump(label_encoder, os.path.join(MODEL_PATH, "label_encoder.pkl"))

print("Model and label encoder saved successfully.")
