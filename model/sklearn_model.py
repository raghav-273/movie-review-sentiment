# model/sklearn_model.py.... saare models hai sklearn based
import os
import joblib
import nltk
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')

class SklearnSentimentModel:
    def __init__(self):
        # pls check and ensure(do baar) movie_reviews corpus is present
        try:
            nltk.data.find('corpora/movie_reviews')
        except LookupError:
            nltk.download('movie_reviews')

        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
        else:
            self.model = self._train_and_save()

    def _train_and_save(self):
        # Load dataset
        documents = []
        labels = []
        for category in movie_reviews.categories():
            for fileid in movie_reviews.fileids(category):
                documents.append(movie_reviews.raw(fileid))
                labels.append(1 if category == 'pos' else 0)

        X_train, X_test, y_train, y_test = train_test_split(
            documents, labels, test_size=0.2, random_state=42
        )

        pipeline = make_pipeline(
            TfidfVectorizer(stop_words='english', max_features=8000),
            LogisticRegression(max_iter=1000)
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"[SklearnSentimentModel] Trained. Test accuracy: {acc:.3f}")
        joblib.dump(pipeline, MODEL_PATH)
        return pipeline

    def predict(self, text: str):
        text = text.lower().strip()
        if not text:
            return "Neutral", 0.0

        probs = self.model.predict_proba([text])[0]  # [p_neg, p_pos]
        pos_prob = float(probs[1])
        neg_prob = float(probs[0])

        # Multi-class decision
        if pos_prob >= 0.6:
            label = "Positive"
        elif pos_prob <= 0.4:
            label = "Negative"
        else:
            label = "Neutral"

        return label, pos_prob