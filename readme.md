Movie Review Sentiment Analysis (Flask + VADER / sklearn)

Description

A simple web-based tool to classify movie reviews as Positive, Negative, or Neutral.
Supports two backends:
	•	VADER (rule-based)
	•	TF-IDF + Logistic Regression (trained on NLTK movie_reviews)

Built with Flask for a clean and fast web interface. Easy to use!

Setup
	1.	python3 -m venv venv  (create a virtual environment)
	2.	source venv/bin/activate  (Windows: .\venv\Scripts\Activate.ps1)
	3.	pip install -r requirements.txt  (install all dependencies)
	4.	python app.py  (start the server)
	5.	Open in browser: http://127.0.0.1:5000

Files
	•	app.py — Main Flask server
	•	model/vader_model.py — VADER wrapper
	•	model/sklearn_model.py — TF-IDF + Logistic Regression and training script
	•	templates/index.html — Frontend
	•	static/style.css — CSS for styling

Notes
	•	Change MODEL_CHOICE in app.py to switch between models.
	•	First run with sklearn will download NLTK corpora and train the model — phir model/sentiment_model.pkl save ho jayega.
	•	VADER works out-of-the-box, no training needed.
