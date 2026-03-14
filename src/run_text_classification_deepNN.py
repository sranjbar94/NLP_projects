# src/run_text_classification_deepNN.py
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Device setup
# ---------------------------
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# ---------------------------
# Load Dataset
# ---------------------------
DATA_PATH = "/Users/bob/Documents/GitHub/NLP_LLM_RAG/NLP_projects/data/raw/news_dataset.json"

print("Loading news dataset...")
df = pd.read_json(DATA_PATH, lines=True)

# Keep only needed columns
df = df[['headline', 'category']].rename(columns={'headline':'text', 'category':'label'})
print("Dataset size:", df.shape)
print("Sample data:\n", df.head())

# ---------------------------
# Encode labels
# ---------------------------
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
num_classes = len(label_encoder.classes_)
print("Number of classes:", num_classes)

# ---------------------------
# Split dataset
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label_encoded'], test_size=0.2, random_state=42, stratify=df['label_encoded']
)
print("Split dataset:", X_train.shape, X_test.shape)

# ---------------------------
# TF-IDF features
# ---------------------------
print("Creating TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Convert to dense tensors
X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_vec.toarray(), dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

# ---------------------------
# DataLoader for batching
# ---------------------------
batch_size = 256
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ---------------------------
# Define Neural Network
# ---------------------------
class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = TextClassifier(input_dim=X_train_tensor.shape[1], hidden_dim=512, num_classes=num_classes)
model.to(device)
print(model)

# ---------------------------
# Loss and Optimizer
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------
# Training Loop
# ---------------------------
epochs = 5
print("Training model...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        # Move batch to device (already on device if tensors are)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# ---------------------------
# Evaluation
# ---------------------------
model.eval()
all_preds = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
y_pred = torch.cat(all_preds).numpy()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ---------------------------
# Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
os.makedirs("../results/figures", exist_ok=True)
plt.savefig("../results/figures/text_classification_deepNN_confusion_matrix.png")
plt.close()
print("Confusion matrix saved to ../results/figures/text_classification_deepNN_confusion_matrix.png")

# ---------------------------
# Save Model
# ---------------------------
os.makedirs("../models/saved_models", exist_ok=True)
torch.save(model.state_dict(), "../models/saved_models/text_classification_deepNN.pth")
print("Model saved to ../models/saved_models/text_classification_deepNN.pth")
