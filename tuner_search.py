import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load data function
def load_data(filename):
    # Updated path to access the data folder from the new directory level
    with open(os.path.join('data', filename), 'r') as f:
        return json.load(f)

# Preprocessing function
def preprocess_text(text):
    return text.lower()

# Load and preprocess the data
train_data = load_data('train.json')
test_data = load_data('test.json')

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

train_df['cleaned_reviews'] = train_df['reviews'].apply(preprocess_text)
test_df['cleaned_reviews'] = test_df['reviews'].apply(preprocess_text)

# Tokenizing the text
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(train_df['cleaned_reviews'])

X_train = tokenizer.texts_to_sequences(train_df['cleaned_reviews'])
X_test = tokenizer.texts_to_sequences(test_df['cleaned_reviews'])

X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

y_train = train_df['sentiments'].values

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build the model for Keras Tuner
def build_model(hp):
    model = Sequential()

    # Embedding Layer
    model.add(Embedding(input_dim=MAX_VOCAB_SIZE, 
                        output_dim=hp.Choice('embedding_dim', values=[50, 100, 200]), 
                        input_length=MAX_SEQUENCE_LENGTH))

    # LSTM Layer
    model.add(LSTM(units=hp.Int('lstm_units', min_value=64, max_value=256, step=64), return_sequences=False))

    # Dropout Layer
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.3, max_value=0.6, step=0.1)))

    # Fully connected layer with output
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Initialize the Keras Tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,  # Number of hyperparameter combinations to try
    executions_per_trial=1,  # Number of models to build and test for each trial
    directory='tuner_dir',  # Directory to save the search results
    project_name='lstm_tuning'
)

# Display the search space summary
tuner.search_space_summary()

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Run the hyperparameter search
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32, callbacks=[early_stopping])

# Retrieve the best model from the tuner
best_model = tuner.get_best_models(num_models=1)[0]

# Summary of the best model
best_model.summary()

# Evaluate the best model on validation data
val_loss, val_accuracy = best_model.evaluate(X_val, y_val)
print(f"Best validation accuracy: {val_accuracy:.4f}")

# Predict on the test data with the best model
test_predictions = (best_model.predict(X_test) > 0.5).astype("int32")

# Save predictions to CSV (updated path to save in data folder)
output_df = test_df.copy()
output_df['predicted_sentiments'] = test_predictions
output_df[['reviews', 'predicted_sentiments']].to_csv('data/submission.csv', index=False)

print("Predictions saved to submission.csv")
