from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Flask app setup

app = Flask(__name__)

# Load and preprocess the dataset
dataset_path = "spam.csv"  # Replace with your dataset path
df = pd.read_csv(dataset_path, encoding='latin-1')

# Data cleaning
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df = df[['label', 'message']]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Vectorize the messages
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    sms = request.form['sms']
    sms_vec = vectorizer.transform([sms])
    prediction = model.predict(sms_vec)
    result = "Spam" if prediction[0] == 1 else "Ham"
    return render_template('index.html', prediction=result, sms=sms)

if __name__ == '__main__':
    app.run(debug=True)
