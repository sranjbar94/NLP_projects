# src/train_deepNN.py
import torch
import torch.nn as nn
from tqdm import tqdm

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        pooled = embedded.mean(dim=1)         # mean pooling
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

def train_model(model, train_loader, val_loader, device='cpu', epochs=5, lr=1e-3):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())
    return model
