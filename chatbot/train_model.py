import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import string
from nltk.tokenize import word_tokenize
import joblib
import nltk  # Add this import

# Download stopwords dataset
nltk.download('stopwords')

# Load the dataset
data = pd.read_csv("static/Conversation.csv")

# Preprocess the text data without removing stopwords
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in string.punctuation]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

data['cleaned_question'] = data['question'].apply(preprocess_text)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['cleaned_question'])
y = data['answer']

# Train an Intent Recognition Model
model = LogisticRegression()
model.fit(X, y)

# Save the model and TF-IDF vectorizer
joblib.dump(model, 'intent_recognition_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_intent', methods=['POST'])
def get_intent():
    user_question = request.form['user_question']

    # Preprocess user's question and transform it using the same TF-IDF vectorizer
    user_question_cleaned = preprocess_text(user_question)
    user_question_vectorized = tfidf_vectorizer.transform([user_question_cleaned])

    # Predict the intent for the user's question
    intent = model.predict(user_question_vectorized)[0]

    return render_template('index.html', user_question=user_question, intent=intent)

if __name__ == '__main__':
    app.run(debug=True)
