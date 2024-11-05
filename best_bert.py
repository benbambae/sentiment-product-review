import os
import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer, 
    BertModel,  # Changed from BertForSequenceClassification to BertModel
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
import torch.nn.functional as F
import re

class TextPreprocessor:
    @staticmethod
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = TextPreprocessor()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.preprocessor.clean_text(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTSentimentClassifier(torch.nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # Use the [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def train_model(model, train_loader, val_loader, device, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=1.69e-5, weight_decay=0.01)
    
    total_steps = len(train_loader) * num_epochs
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    best_val_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Average training loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_accuracy = evaluate_model(model, val_loader, device)
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with validation accuracy: {val_accuracy:.4f}")

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.tolist())
    
    return accuracy_score(actual_labels, predictions)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        train_data = pd.read_json(os.path.join('data', 'train.json'))
        test_data = pd.read_json(os.path.join('data', 'test.json'))
        print(f"Data loaded successfully. Training samples: {len(train_data)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    X_train, X_val, y_train, y_val = train_test_split(
        train_data['reviews'],
        train_data['sentiments'],
        test_size=0.2,
        random_state=42,
        stratify=train_data['sentiments']
    )
    
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTSentimentClassifier().to(device)
    print("Model initialized")
    
    train_dataset = SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    val_dataset = SentimentDataset(X_val.tolist(), y_val.tolist(), tokenizer)
    test_dataset = SentimentDataset(test_data['reviews'].tolist(), [0]*len(test_data), tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    print("Starting training...")
    train_model(model, train_loader, val_loader, device, num_epochs=4)
    
    print("Loading best model for predictions...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    print("Making predictions on test set...")
    model.eval()
    test_predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            _, preds = torch.max(outputs, dim=1)
            test_predictions.extend(preds.cpu().tolist())
    
    test_data['predicted_sentiments'] = test_predictions
    output_file = './data/improved_bert_submission.csv'
    test_data[['reviews', 'predicted_sentiments']].to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()