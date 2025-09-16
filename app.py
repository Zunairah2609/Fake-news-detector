from flask import Flask, render_template, request
import joblib

# Create Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

@app.route('/')
def home():
    return render_template('detectorindex.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form['news']
    news_vectorized = vectorizer.transform([news_text])
    prediction = model.predict(news_vectorized)[0]

    result = "Fake" if prediction == 0 else "Real"
    return render_template('detectorindex.html', prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
