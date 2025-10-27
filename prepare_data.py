import pandas as pd
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load CSV
df = pd.read_csv('data/IMDB_Dataset.csv')

# Check columns
print(df.columns)  # should show ['review', 'sentiment']

# Encode sentiment labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment'])  # positive=1, negative=0

# Clean review text
df['review'] = df['review'].str.replace('<br />', ' ', regex=False).str.lower()
df['review'] = df['review'].str.translate(str.maketrans('', '', string.punctuation))

# Split full dataset (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['label'], test_size=0.2, random_state=42
)

print("Preprocessing done!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Optional: save cleaned full dataset
df.to_csv('data/imdb_cleaned.csv', index=False)