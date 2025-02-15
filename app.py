from flask import Flask, render_template, request
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = load_model("notebook\Rnn_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    word_index = imdb.get_word_index()
    word_indices = [word_index.get(i, -1) for i in review.split()]
    padded_sequence = pad_sequences([word_indices], padding='post', maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
