from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load model
model, vectorizer = pickle.load(open('sentiment_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    sentiment = 'Positive 😊' if prediction == 1 else 'Negative 😞'
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)

