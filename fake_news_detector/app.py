import csv
from pathlib import Path

from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fact_check import verify_online, combined_prediction


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "news.csv"


def create_model():
    """Train a small text classifier from the included example dataset."""
    texts = []
    labels = []

    with DATA_PATH.open(encoding="utf-8", newline="") as dataset:
        for row in csv.DictReader(dataset):
            texts.append(row["text"])
            labels.append(row["label"])

    if len(texts) < 4 or len(set(labels)) < 2:
        raise ValueError("The dataset must contain at least four examples from two labels.")

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(texts, labels)
    return model


model = create_model()
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    article_text = ""

    if request.method == "POST":
        article_text = request.form.get("article_text", "").strip()
        if article_text:
            # 1) ML model prediction
            ml_prediction = model.predict([article_text])[0]
            probabilities = model.predict_proba([article_text])[0]
            ml_confidence = max(probabilities) * 100

            # 2) Free internet verification (no API key required)
            online = verify_online(article_text)

            # 3) Combine ML + internet verification
            result = combined_prediction(ml_prediction, ml_confidence, online)

    return render_template("index.html", result=result, article_text=article_text)


if __name__ == "__main__":
    app.run(debug=True)
