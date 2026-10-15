from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load your existing trained model and vectorizer
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "fake_news_model.pkl")
tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")

app = FastAPI(title="TruthLens API")

# Allow your Lovable frontend to call this API.
# We can restrict this to your frontend domain later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewsRequest(BaseModel):
    article: str = Field(min_length=1, max_length=200000)


@app.get("/")
def home():
    return {"message": "TruthLens API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_news(request: NewsRequest):
    if not request.article.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a news article."
        )

    # Convert the article into TF-IDF features
    article_tfidf = tfidf.transform([request.article])

    # Predict the class
    prediction = int(model.predict(article_tfidf)[0])

    # Get estimated class probabilities
    probabilities = model.predict_proba(article_tfidf)[0]

    # Map probabilities to their corresponding labels
    probability_map = {
        int(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }

    return {
        "prediction": "FAKE" if prediction == 0 else "REAL",
        "fake_probability": round(probability_map[0], 4),
        "real_probability": round(probability_map[1], 4)
    }
