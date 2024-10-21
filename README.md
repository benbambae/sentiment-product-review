# Sentiment Analysis of Product Reviews

This project is part of the IE4483 Artificial Intelligence and Data Mining course. The goal is to classify product reviews as either positive or negative using sentiment analysis techniques.

## Project Overview

The project reads product reviews from JSON files, processes the text (removing common words, converting text to numeric features using TF-IDF), trains a Logistic Regression classifier, and predicts whether reviews in the test set are positive or negative. The results are saved to a CSV file.

## Requirements

The project requires the following dependencies:

- Python 3.12+
- pandas
- scikit-learn
- nltk

You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
