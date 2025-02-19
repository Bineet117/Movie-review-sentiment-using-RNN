from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

app = Flask(__name__)

# Load the pre-trained model once
model = load_model("notebook/Rnn_model.h5")

def processing_input(text):
    """Convert text into word indices using IMDB's built-in word index and pad the sequence."""
    word_index = imdb.get_word_index()
    word_indices = [word_index.get(word, 0) + 3 for word in text.split()]  # Offset by 3 (IMDB reserves 0,1,2)
    padded_sequence = pad_sequences([word_indices], padding='post', maxlen=200)
    return padded_sequence

def predict_sentiment(text):
    """Predict sentiment for the processed text."""
    padded_text = processing_input(text)
    prediction = model.predict(padded_text)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    sentiment, confidence = predict_sentiment(review)
    return render_template('index.html', review=review, sentiment=sentiment, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
