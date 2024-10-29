import os
import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import optuna  # Import Optuna for hyperparameter optimization

# Load Data
def load_data(filename):
    with open(os.path.join('data', filename), 'r') as f:
        return json.load(f)

# Prepare Data
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {**{k: v.squeeze() for k, v in encoding.items()}, 'label': torch.tensor(label, dtype=torch.long)}

# Load and preprocess data
train_data = load_data('train.json')
test_data = load_data('test.json')

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Split data for BERT
X_train, X_val, y_train, y_val = train_test_split(train_df['reviews'], train_df['sentiments'], test_size=0.2, random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create Dataset objects
train_dataset = SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer)
val_dataset = SentimentDataset(X_val.tolist(), y_val.tolist(), tokenizer)
test_dataset = SentimentDataset(test_df['reviews'].tolist(), [0]*len(test_df), tokenizer)  # Use dummy labels for test set

# Data Collator for padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define compute_metrics for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Define model_init for Trainer with Optuna
def model_init():
    return BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Define the hyperparameter search space
def hyperparameter_search_fn(trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 5e-5),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16])
    }

# Trainer setup with hyperparameter search
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    weight_decay=0.01,
    load_best_model_at_end=True
)

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Run hyperparameter search with Optuna
best_run = trainer.hyperparameter_search(
    direction="maximize",
    hp_space=hyperparameter_search_fn,
    backend="optuna"
)

print("Best hyperparameters:", best_run.hyperparameters)

# Train the model with best hyperparameters
trainer.train()

# Evaluate the model on validation data
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Predict on the test data
predictions = trainer.predict(test_dataset)
test_df['predicted_sentiments'] = predictions.predictions.argmax(axis=1)

# Save predictions to CSV
test_df[['reviews', 'predicted_sentiments']].to_csv('./data/bert_submission.csv', index=False)
print("Predictions saved to bert_submission.csv")
