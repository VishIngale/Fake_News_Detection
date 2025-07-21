import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Changed for probability support
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
fake = pd.read_csv("C:/Users/BHAUSAHEB/Downloads/Fake.csv")
true = pd.read_csv("C:/Users/BHAUSAHEB/Downloads/True.csv")

# Add labels
fake['label'] = 0  # Fake
true['label'] = 1  # Real

# Keep only 'text' and 'label'
data = pd.concat([fake[['text', 'label']], true[['text', 'label']]])
data.dropna(inplace=True)
data['text'] = data['text'].str.strip()

# Shuffle the data
data = data.sample(frac=1, random_state=42)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Logistic Regression model (supports predict_proba)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vect, y_train)

# Evaluate
y_pred = model.predict(X_test_vect)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/tfidf_vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved successfully!")
