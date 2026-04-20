# src/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_name='distilbert-base-uncased', max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_dataloaders(df, label_column='label', text_column='text', batch_size=32, max_len=128):
    le = LabelEncoder()
    df[label_column] = le.fit_transform(df[label_column])
    
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[label_column])
    
    train_dataset = NewsDataset(train_df[text_column].tolist(), train_df[label_column].tolist(), max_len=max_len)
    val_dataset = NewsDataset(val_df[text_column].tolist(), val_df[label_column].tolist(), max_len=max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, le
