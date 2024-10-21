import json
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from nltk.corpus import stopwords
import nltk
import ssl

# Fix SSL certificate issue for NLTK download
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download stopwords if you haven't already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define the path to the data folder
DATA_FOLDER = 'data/'

# Function to load the JSON files
def load_data(filename):
    with open(os.path.join(DATA_FOLDER, filename), 'r') as f:
        return json.load(f)

# Preprocessing function to clean the text
def preprocess_text(text):
    # Convert to lowercase and remove stopwords
    words = text.lower().split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Load the training and testing data
train_data = load_data('train.json')
test_data = load_data('test.json')

# Convert the data to Pandas DataFrames
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Preprocess the reviews
train_df['cleaned_reviews'] = train_df['reviews'].apply(preprocess_text)
test_df['cleaned_reviews'] = test_df['reviews'].apply(preprocess_text)

# TF-IDF Vectorizer to convert text to numeric features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the training data
X_train = tfidf_vectorizer.fit_transform(train_df['cleaned_reviews'])
y_train = train_df['sentiments']

# Transform the test data
X_test = tfidf_vectorizer.transform(test_df['cleaned_reviews'])

# Simple Logistic Regression model (you can replace this with any model)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict on the test data
test_predictions = classifier.predict(X_test)

# Save predictions to CSV
output_df = test_df.copy()
output_df['predicted_sentiments'] = test_predictions
output_df[['reviews', 'predicted_sentiments']].to_csv('submission.csv', index=False)

print("Predictions saved to submission.csv")
