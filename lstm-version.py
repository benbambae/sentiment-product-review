import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load data function
def load_data(filename):
    with open(os.path.join('data', filename), 'r') as f:
        return json.load(f)

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    return text.lower()

# Load the data
train_data = load_data('train.json')
test_data = load_data('test.json')

# Convert to DataFrame
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Preprocess the reviews
train_df['cleaned_reviews'] = train_df['reviews'].apply(preprocess_text)
test_df['cleaned_reviews'] = test_df['reviews'].apply(preprocess_text)

# Tokenizing the text
MAX_VOCAB_SIZE = 10000  # Limit on the number of words to include in the vocabulary
MAX_SEQUENCE_LENGTH = 100  # Maximum length of each review (padded/truncated)

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(train_df['cleaned_reviews'])

# Convert the text to sequences
X_train = tokenizer.texts_to_sequences(train_df['cleaned_reviews'])
X_test = tokenizer.texts_to_sequences(test_df['cleaned_reviews'])

# Pad the sequences to make sure they are all the same length
X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

# Extract the labels
y_train = train_df['sentiments'].values

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()

# Embedding layer (turns words into dense vectors)
EMBEDDING_DIM = 100
model.add(Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))

# LSTM layer
model.add(LSTM(128, return_sequences=False))

# Dropout layer to prevent overfitting
model.add(Dropout(0.5))

# Fully connected layer with output (sigmoid for binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model architecture
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# Predict on the test data
test_predictions = (model.predict(X_test) > 0.5).astype("int32")

# Save predictions to CSV
output_df = test_df.copy()
output_df['predicted_sentiments'] = test_predictions
output_df[['reviews', 'predicted_sentiments']].to_csv('submission.csv', index=False)

print("Predictions saved to submission.csv")
