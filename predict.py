import joblib

# Load model and vectorizer files we just saved
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

def predict_news(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    # show confidence for LogisticRegression if available
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0]
        conf = max(prob)
        label = "Real" if pred == 1 else "Fake"
        return f"{label} (confidence {conf:.2f})"
    else:
        return "Real" if pred == 1 else "Fake"

if __name__ == "__main__":
    print("Loaded model. Type 'quit' to exit.")
    while True:
        text = input("\nEnter news text: ").strip()
        if text.lower() == "quit":
            print("Goodbye")
            break
        if text == "":
            print("Please type something (or 'quit').")
            continue
        print("Prediction:", predict_news(text))
