import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import nltk
import re
import string
import unicodedata
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
from collections import Counter

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load the dataset
IMDB_DATA_PATH = '/Users/amninderrandhawa/Downloads/IMDB Dataset.csv'
imdb_data = pd.read_csv(IMDB_DATA_PATH)

# Function to clean and preprocess text
def preprocess_text(text):
    # Remove HTML tags
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    
    # Remove square brackets and contents inside them
    text = re.sub('\[[^]]*\]', '', text)
    
    # Remove accented characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Tokenize text
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    
    # Remove stopwords
    stopword_list = stopwords.words('english')
    tokens = [token.strip() for token in tokens if token.lower() not in stopword_list]
    
    # Stemming
    porter_stemmer = PorterStemmer()
    tokens = [porter_stemmer.stem(token) for token in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    text = ' '.join(tokens)
    
    return text

# Apply text preprocessing to the 'review' column
imdb_data['clean_review'] = imdb_data['review'].apply(preprocess_text)

# Split the dataset into train and test
train_data = imdb_data.iloc[:40000]
test_data = imdb_data.iloc[40000:]

# Vectorize text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(min_df=1, max_df=1, use_idf=True, ngram_range=(1, 3))
X_train = tfidf_vectorizer.fit_transform(train_data['clean_review'])
X_test = tfidf_vectorizer.transform(test_data['clean_review'])

# Label encoding for sentiments
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(train_data['sentiment'])
y_test = label_binarizer.transform(test_data['sentiment'])

# Train logistic regression model
logistic_regression = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
logistic_regression.fit(X_train, y_train)

# Predictions
y_pred = logistic_regression.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Tokenize positive and negative reviews and remove symbols
positive_tokens = [word for word in ' '.join(train_data[train_data['sentiment'] == 'positive']['clean_review']).split() if word not in string.punctuation]
negative_tokens = [word for word in ' '.join(train_data[train_data['sentiment'] == 'negative']['clean_review']).split() if word not in string.punctuation]

# Count the occurrence of each word
positive_word_counts = Counter(positive_tokens)
negative_word_counts = Counter(negative_tokens)

# Sort word counts in descending order
positive_word_counts_sorted = sorted(positive_word_counts.items(), key=lambda x: x[1], reverse=True)
negative_word_counts_sorted = sorted(negative_word_counts.items(), key=lambda x: x[1], reverse=True)

# Extract top 30 words and their counts
top_positive_words = positive_word_counts_sorted[:30]
top_negative_words = negative_word_counts_sorted[:30]

# Convert to pandas DataFrame
positive_word_df = pd.DataFrame(top_positive_words, columns=['Word', 'Count'])
negative_word_df = pd.DataFrame(top_negative_words, columns=['Word', 'Count'])

# Print the top 30 words for each sentiment
print("Top 30 Positive Words:")
print(positive_word_df)
print("\nTop 30 Negative Words:")
print(negative_word_df)