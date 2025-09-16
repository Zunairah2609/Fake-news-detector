import pandas as pd

# Step 1: Load the data
fake_df = pd.read_csv("Data/Fake.csv")
true_df = pd.read_csv("Data/True.csv")

# Step 2: Label the data
fake_df["label"] = 0
true_df["label"] = 1

# Step 3: Combine both
df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows

print("Dataset Loaded")
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 4: Split into training and testing
X = df['text']   # our news text
y = df['label']  # fake=0, real=1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 7: Test the model
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

# Step 8: Prediction function
def predict_news(text):
    text_tfidf = vectorizer.transform([text])
    pred = model.predict(text_tfidf)
    return "Real" if pred[0] == 1 else "Fake"

# Try with your own input
print(predict_news("Donald Trump says the moon is made of cheese"))

import joblib

# Save the trained model and the TF-IDF vectorizer
joblib.dump(model, "model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print("Saved model -> model.joblib")
print("Saved vectorizer -> vectorizer.joblib")


