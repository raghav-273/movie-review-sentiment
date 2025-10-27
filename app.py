# app.py
from flask import Flask, render_template, request, jsonify
import pickle
from wordcloud import WordCloud
from io import BytesIO
import base64
import pandas as pd
import string
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# ---------- Load trained model + vectorizer ----------
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

# ---------- Helper: Word cloud ----------
def generate_wordcloud(text):
    wc = WordCloud(width=400, height=200, background_color="white").generate(text)
    buffer = BytesIO()
    wc.to_image().save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64," + img_str

# ---------- Helper: Predict sentiment from model ----------
def predict_sentiment(text):
    X_vec = vectorizer.transform([text])
    probs = model.predict_proba(X_vec)[0]  # [neg_prob, pos_prob]

    # Ensure numeric
    neg_prob = float(probs[0])
    pos_prob = float(probs[1])
    neutral_prob = max(0.0, 1.0 - (neg_prob + pos_prob))

    # Multi-class thresholding
    if pos_prob >= 0.6:
        label = "Positive"
    elif pos_prob <= 0.4:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "label": label,
        "prob_pos": pos_prob,
        "prob_neg": neg_prob,
        "prob_neutral": neutral_prob
    }

# ---------- Compute performance metrics once at startup ----------
def compute_metrics():
    try:
        df = pd.read_csv('data/imdb_cleaned.csv')  # cleaned dataset created earlier
        df['review'] = df['review'].astype(str).str.replace('<br />', ' ', regex=False).str.lower()
        df['review'] = df['review'].str.translate(str.maketrans('', '', string.punctuation))
        df['label'] = (df['sentiment'] == 'positive').astype(int)

        X_train, X_test, y_train, y_test = train_test_split(df['review'], df['label'], test_size=0.2, random_state=42)
        X_test_vec = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)

        return {
            'accuracy': round(float(acc), 4),
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1': round(float(f1), 4)
        }
    except Exception as e:
        print("Warning: could not compute metrics at startup:", e)
        return { 'accuracy': None, 'precision': None, 'recall': None, 'f1': None }

precomputed_metrics = compute_metrics()
print("Precomputed metrics:", precomputed_metrics)

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('index.html', model_choice='SKLEARN', metrics=precomputed_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('review', '').strip()
    if not text:
        return jsonify({
            'label': "Neutral",
            'score': 0.0,
            'prob_pos': 0.0,
            'prob_neg': 0.0,
            'prob_neutral': 0.0,
            'wordcloud': '',
            'metrics': precomputed_metrics
        })

    # Get prediction
    result = predict_sentiment(text)
    wc_img = generate_wordcloud(text)

    return jsonify({
        'label': result['label'],
        'score': result['prob_pos'],
        'prob_pos': result['prob_pos'],
        'prob_neg': result['prob_neg'],
        'prob_neutral': result['prob_neutral'],
        'wordcloud': wc_img,
        'metrics': precomputed_metrics
    })

if __name__ == "__main__":
    app.run(debug=True)