import pickle
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load the data
data = pd.read_csv("BBC News Train.csv")

# Text preprocessing function
def process_text(text):
    text = text.lower().replace('\n',' ').replace('\r',' ').strip()
    text = re.sub(' +',' ',text)
    text = re.sub(r'[^\w\s]','',text)
    
    # Removing stop words
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
            
    text = ' '.join(filtered_sentence)
    return text

# Preprocess the text data
data["Text_parsed"] = data['Text'].apply(process_text)

# Encode the target labels
label_encoder = preprocessing.LabelEncoder()
data['Category_target'] = label_encoder.fit_transform(data['Category'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    data['Text_parsed'],
    data['Category_target'],
    test_size=0.2,
    random_state=8
)

# Create and fit TF-IDF vectorizer
tfidf = TfidfVectorizer(
    encoding='utf-8',
    ngram_range=(1, 2),
    stop_words=None,
    lowercase=False,
    max_df=0.50,
    min_df=5,
    max_features=300,
    norm='l2',
    sublinear_tf=True
)

features_train = tfidf.fit_transform(X_train).toarray()
features_test = tfidf.transform(X_test).toarray()

# Train the model
model_rfc = RandomForestClassifier()
model_rfc.fit(features_train, y_train)

# Save the trained model components
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('news_classifier_model.pkl', 'wb') as f:
    pickle.dump(model_rfc, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model components saved successfully!")

# Test the saved model (optional)
test_article = "'England need a lot of work' - reaction and your views after wins over Latvia & Albania"
processed_article = process_text(test_article)
test_features = tfidf.transform([processed_article]).toarray()
prediction = model_rfc.predict(test_features)[0]
category_name = label_encoder.inverse_transform([prediction])[0]

print(f"Test article classified as: {category_name}")