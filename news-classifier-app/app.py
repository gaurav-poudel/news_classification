import pandas as pd
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flask import Flask, render_template, request, jsonify

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

app = Flask(__name__)

# Load the trained model and vectorizer
def load_model():
    try:
        # Load the TF-IDF vectorizer
        tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        
        # Load the trained RandomForest model
        model = pickle.load(open('news_classifier_model.pkl', 'rb'))
        
        # Load the label encoder for category names
        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        
        return tfidf, model, label_encoder
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

# Text preprocessing function (same as used during training)
def process_text(text):
    text = text.lower().replace('\n', ' ').replace('\r', ' ').strip()
    text = re.sub(' +', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
            
    text = ' '.join(filtered_sentence)
    return text

# Load the model and vectorizer
tfidf_vectorizer, classifier_model, label_encoder = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_news():
    if request.method == 'POST':
        if request.is_json:
            # Handle API request
            article_text = request.json.get('text', '')
        else:
            # Handle form submission
            article_text = request.form.get('article_text', '')
        
        if not article_text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess the text
        processed_text = process_text(article_text)
        
        # Transform the text using the vectorizer
        features = tfidf_vectorizer.transform([processed_text]).toarray()
        
        # Make prediction
        prediction = classifier_model.predict(features)[0]
        
        # Get the category name
        category_name = label_encoder.inverse_transform([prediction])[0]
        
        if request.is_json:
            return jsonify({
                'category': category_name,
                'category_id': int(prediction)
            })
        else:
            return render_template('result.html', 
                                  article_text=article_text,
                                  category=category_name)

if __name__ == '__main__':
    app.run(debug=True)