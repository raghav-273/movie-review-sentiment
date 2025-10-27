# vader_model.py ... model hai for vader
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

class VaderSentimentModel:
    def __init__(self):
        # check lexicon if its availabe ...pls download if not 
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        self.sid = SentimentIntensityAnalyzer()

    def predict(self, text: str):
        if not text or text.strip() == "":
            return "Neutral", 0.0
        scores = self.sid.polarity_scores(text)
        compound = scores['compound']  # -1 ya fir 1
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        confidence = (abs(compound))  # 0..1 as strength
        return label, confidence