import pandas as pd
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- Step 1: Load cleaned CSV ---
df = pd.read_csv('data/imdb_cleaned.csv')

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])

# Clean text (just in case)
df['review'] = df['review'].str.replace('<br />', ' ', regex=False).str.lower()
df['review'] = df['review'].str.translate(str.maketrans('', '', string.punctuation))

# Split dataset 80/20
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['label'], test_size=0.2, random_state=42
)

# --- Step 2: Vectorization ---
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- Step 3: Train model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# --- Step 4: Evaluate ---
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Negative','Positive']))

# --- Step 5: Save model and vectorizer ---
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")