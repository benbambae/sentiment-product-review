import os
import json
import pandas as pd
import torch
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
from datetime import datetime
import numpy as np

# Set up logging
logging.basicConfig(
    filename='stats.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    logging.warning("NLTK download failed, using basic tokenization")
    pass

class PredictionAnalyzer:
    def __init__(self, true_labels, predicted_labels, texts, threshold=0.5):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.texts = texts
        self.threshold = threshold
        self.confusion_dict = self._analyze_predictions()
    
    def _analyze_predictions(self):
        results = {
            'true_positives': [],
            'true_negatives': [],
            'false_positives': [],
            'false_negatives': []
        }
        
        for i, (true, pred, text) in enumerate(zip(self.true_labels, self.predicted_labels, self.texts)):
            if true == 1 and pred == 1:
                results['true_positives'].append((i, text))
            elif true == 0 and pred == 0:
                results['true_negatives'].append((i, text))
            elif true == 0 and pred == 1:
                results['false_positives'].append((i, text))
            elif true == 1 and pred == 0:
                results['false_negatives'].append((i, text))
        
        return results
    
    def save_analysis(self, output_dir='analysis_results'):
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each category to separate files
        for category, items in self.confusion_dict.items():
            if items:  # Only save if there are items
                df = pd.DataFrame(items, columns=['index', 'text'])
                df.to_csv(f'{output_dir}/{category}.csv', index=False)
        
        # Log counts
        logging.info("Prediction Analysis Counts:")
        for category, items in self.confusion_dict.items():
            logging.info(f"{category}: {len(items)}")

    def get_metrics(self):
        tn, fp, fn, tp = confusion_matrix(self.true_labels, self.predicted_labels).ravel()
        
        metrics = {
            'accuracy': accuracy_score(self.true_labels, self.predicted_labels),
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        }
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        return metrics

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.preprocess_text(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {**{k: v.squeeze() for k, v in encoding.items()}, 'label': torch.tensor(label, dtype=torch.long)}

    @staticmethod
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        return ' '.join(tokens)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    metrics = {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn
    }
    
    # Log metrics
    logging.info("Evaluation Metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value}")
    
    return metrics

def get_serializable_config(training_args):
    """Convert training arguments to a JSON-serializable format"""
    config = {
        'learning_rate': training_args.learning_rate,
        'per_device_train_batch_size': training_args.per_device_train_batch_size,
        'per_device_eval_batch_size': training_args.per_device_eval_batch_size,
        'num_train_epochs': training_args.num_train_epochs,
        'weight_decay': training_args.weight_decay,
        'output_dir': training_args.output_dir,
        'evaluation_strategy': training_args.evaluation_strategy,
        'save_strategy': training_args.save_strategy,
        'logging_dir': training_args.logging_dir
    }
    return config

def train_and_evaluate():
    # Log start time and system info
    logging.info("="*50)
    logging.info("Starting new training run")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load data
    try:
        with open(os.path.join('data', 'train.json'), 'r') as f:
            train_data = json.load(f)
        with open(os.path.join('data', 'test.json'), 'r') as f:
            test_data = json.load(f)
        logging.info(f"Data loaded successfully. Training samples: {len(train_data)}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return
    
    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        train_df['reviews'],
        train_df['sentiments'],
        test_size=0.2,
        random_state=42,
        stratify=train_df['sentiments']
    )
    
    # Log dataset sizes
    logging.info(f"Training set size: {len(X_train)}")
    logging.info(f"Validation set size: {len(X_val)}")
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    # Create datasets
    train_dataset = SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    val_dataset = SentimentDataset(X_val.tolist(), y_val.tolist(), tokenizer)
    test_dataset = SentimentDataset(test_df['reviews'].tolist(), [0]*len(test_df), tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        learning_rate=1.69e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True
    )
    
    # Log training parameters
    logging.info("Training Parameters:")
    for key, value in training_args.__dict__.items():
        logging.info(f"{key}: {value}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )
    
    # Train model
    trainer.train()
    
    # Final evaluation
    eval_results = trainer.evaluate()
    logging.info("Final Evaluation Results:")
    logging.info(eval_results)
    
    # Make predictions on validation set for detailed analysis
    val_predictions = trainer.predict(val_dataset)
    val_preds = val_predictions.predictions.argmax(-1)
    
    # Create prediction analyzer
    analyzer = PredictionAnalyzer(y_val.tolist(), val_preds, X_val.tolist())
    
    # Save detailed analysis
    analyzer.save_analysis()
    
    # Get and log detailed metrics
    metrics = analyzer.get_metrics()
    logging.info("Detailed Metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value}")
    
    # Make predictions on test set
    test_predictions = trainer.predict(test_dataset)
    test_df['predicted_sentiments'] = test_predictions.predictions.argmax(-1)
    
    # Save predictions
    output_file = './data/bert_submission.csv'
    test_df[['reviews', 'predicted_sentiments']].to_csv(output_file, index=False)
    logging.info(f"Predictions saved to {output_file}")
    
    # Save model configuration
    model_config = {
        'model_name': 'bert-base-uncased',
        'max_length': 128,
        'training_parameters': get_serializable_config(training_args),
        'final_metrics': {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'true_positives': int(metrics['true_positives']),
            'true_negatives': int(metrics['true_negatives']),
            'false_positives': int(metrics['false_positives']),
            'false_negatives': int(metrics['false_negatives'])
        },
        'training_time': str(datetime.now()),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Save configuration
    with open('model_config.json', 'w') as f:
        json.dump(model_config, f, indent=4)
    logging.info("Model configuration saved to model_config.json")
    
    # Save a summary report
    summary = f"""
    =================================
    BERT Model Training Summary
    =================================
    Time: {datetime.now()}
    Model: bert-base-uncased
    Device: {'cuda' if torch.cuda.is_available() else 'cpu'}
    
    Training Parameters:
    - Learning Rate: {training_args.learning_rate}
    - Batch Size: {training_args.per_device_train_batch_size}
    - Epochs: {training_args.num_train_epochs}
    - Weight Decay: {training_args.weight_decay}
    
    Final Metrics:
    - Accuracy: {metrics['accuracy']:.4f}
    - Precision: {metrics['precision']:.4f}
    - Recall: {metrics['recall']:.4f}
    - F1 Score: {metrics['f1_score']:.4f}
    
    Confusion Matrix:
    - True Positives: {int(metrics['true_positives'])}
    - True Negatives: {int(metrics['true_negatives'])}
    - False Positives: {int(metrics['false_positives'])}
    - False Negatives: {int(metrics['false_negatives'])}
    =================================
    """
    
    # Save summary to a file
    with open('training_summary.txt', 'w') as f:
        f.write(summary)
    logging.info("Training summary saved to training_summary.txt")
    
    # Log summary
    logging.info(summary)
    
    return metrics

if __name__ == "__main__":
    train_and_evaluate()