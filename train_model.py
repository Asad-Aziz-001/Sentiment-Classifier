import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# 1. Create a simple dataset
data = {
    'text': [
        'I love this product', 'This is amazing', 'I am very happy',
        'I hate this', 'This is terrible', 'I am sad'
    ],
    'label': [1, 1, 1, 0, 0, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# 2. Text vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 3. Model training
model = LogisticRegression()
model.fit(X, y)

# 4. Save model + vectorizer
with open('model/sentiment_model.pkl', 'wb') as f:
    pickle.dump((model, vectorizer), f)

print("✅ Model trained and saved successfully!")
