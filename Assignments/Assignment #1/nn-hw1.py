# Python Version: 3.12.5

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import numpy as np
import nltk
import re
import contractions
import warnings

warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Configuration
class ConfigValues:
    RANDOM_STATE_VALUE = 42
    MAX_TFIDF_FEATURES = 45000

# Load dataset
url = "https://web.archive.org/web/20201127142707if_/https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Office_Products_v1_00.tsv.gz"
data = pd.read_csv(url, sep='\t', compression='gzip', on_bad_lines='skip')

data = data[['review_body', 'star_rating']].dropna()
data['star_rating'] = pd.to_numeric(data['star_rating'], errors='coerce')

data = data.dropna()
neutral_count = (data['star_rating'] == 3).sum()

# Sentiment Mapping
data = data[data['star_rating'] != 3]
data['sentiment'] = data['star_rating'].apply(lambda x: 1 if x > 3 else 0)

# Print review statistics
positive_count = (data['sentiment'] == 1).sum()
negative_count = (data['sentiment'] == 0).sum()

print(f"Positive reviews: {positive_count}")
print(f"Negative reviews: {negative_count}")
print(f"Neutral reviews (discarded): {neutral_count}")

# Downsize dataset
positive_reviews = data[data['sentiment'] == 1].sample(100000, random_state=ConfigValues.RANDOM_STATE_VALUE)
negative_reviews = data[data['sentiment'] == 0].sample(100000, random_state=ConfigValues.RANDOM_STATE_VALUE)

dataset = pd.concat([positive_reviews, negative_reviews]).sample(frac=1, random_state=ConfigValues.RANDOM_STATE_VALUE)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(dataset['review_body'], dataset['sentiment'], test_size=0.2, random_state=ConfigValues.RANDOM_STATE_VALUE)

# Data Cleaning
def clean_text(text):
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

x_train = x_train.apply(clean_text)
x_test = x_test.apply(clean_text)

# Length calculations
avg_length_before = dataset['review_body'].str.len().mean()
avg_length_after = x_train.str.len().mean()
print(f"Average length before cleaning: {avg_length_before:.4f}")
print(f"Average length after cleaning: {avg_length_after:.4f}")

# Pre-processing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

x_train = x_train.apply(preprocess_text)
x_test = x_test.apply(preprocess_text)

# avg Length after processing
avg_length_after_processing = x_train.str.len().mean()
print(f"Average length before pre-processing: {avg_length_after:.4f}")
print(f"Average length after preprocessing: {avg_length_after_processing:.4f}")

# TF-IDF
vectorizer = TfidfVectorizer(max_features=ConfigValues.MAX_TFIDF_FEATURES)
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

# Train Models and Print Metrics
def evaluate_model(model, param_grid, name):
    # model.fit(x_train_tfidf, y_train)
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    grid_search.fit(x_train_tfidf, y_train)
    best_model = grid_search.best_estimator_

    train_pred = best_model.predict(x_train_tfidf)
    test_pred = best_model.predict(x_test_tfidf)
    
    print(f"{name}: {accuracy_score(y_train, train_pred):.4f}, {precision_score(y_train, train_pred):.4f}, {recall_score(y_train, train_pred):.4f}, {f1_score(y_train, train_pred):.4f}, {accuracy_score(y_test, test_pred):.4f}, {precision_score(y_test, test_pred):.4f}, {recall_score(y_test, test_pred):.4f}, {f1_score(y_test, test_pred):.4f}")

# Param grids for tuning
param_grid_perceptron = {
    'penalty': ['l2', 'elasticnet'],
    'alpha': [0.00005, 0.0001, 0.001, 0.005],
    'max_iter': [2000, 3000, 5000, 7000]
}

param_grid_svc = {
    'C': [0.1, 1, 10],
    'max_iter': [1000, 3000],
    'loss': ['squared_hinge'],
}

param_grid_nb = {
    'alpha': [0.01, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0, 10.0]
}

param_grid_lr = {

}

# Models
evaluate_model(Perceptron(max_iter=3000), param_grid_perceptron, "Perceptron")
evaluate_model(LinearSVC(max_iter=3000), param_grid_svc, "LinearSVC")
evaluate_model(LogisticRegression(), param_grid_lr, "Logistic Regression")
evaluate_model(MultinomialNB(), param_grid_nb, "Naive Bayes")
