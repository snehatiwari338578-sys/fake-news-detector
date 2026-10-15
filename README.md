# 📰 TruthLens — Fake News Detection

An end-to-end Machine Learning project that classifies news articles as Fake or Real using Natural Language Processing (NLP) and supervised learning.

## 🚀 Live Demo

**Frontend:** [Open TruthLens](https://intel-scan.lovable.app/)

**Backend API:** [https://fake-news-detector-b7pz.onrender.com](https://fake-news-detector-b7pz.onrender.com)

**API Documentation:** [https://fake-news-detector-b7pz.onrender.com/docs](https://fake-news-detector-b7pz.onrender.com/docs)

## ✨ Features

- News article classification using a trained ML model.
- TF-IDF text vectorization.
- Logistic Regression prediction.
- Estimated probabilities for Fake and Real classes.
- Interactive interface with loading states and error handling.
- REST API built with FastAPI.
- Deployed frontend and Python backend.

## 🛠️ Tech Stack

**Frontend:** Lovable, React

**Backend:** Python, FastAPI, Uvicorn

**Machine Learning:** Scikit-learn, TF-IDF, Logistic Regression, Multinomial Naive Bayes

**Data Processing:** Pandas, NumPy

**Deployment:** Lovable, Render

## ⚙️ How It Works

1. The user enters a news article in the frontend.
2. The frontend sends the article to the FastAPI `/predict` endpoint.
3. The saved TF-IDF vectorizer converts the article into numerical features.
4. The trained Logistic Regression model predicts the class.
5. The backend returns the prediction and estimated class probabilities.
6. The frontend displays the result.

## 📊 Model Evaluation

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 98.62% |
| Multinomial Naive Bayes | 93.13% |

These results were measured on a held-out test split of the dataset. They should not be interpreted as real-world fact-checking accuracy.

## 🧠 Dataset and Methodology

The project uses the Fake and Real News Dataset. The dataset is combined, cleaned, and split into training and testing sets. TF-IDF features are then used to train and evaluate classification models.

## ▶️ Run the Backend Locally

Install the dependencies:

`pip install -r requirements.txt`

Start the API from the directory containing `api.py` and the saved model files:

`uvicorn api:app --reload`

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

## ⚠️ Disclaimer

TruthLens performs machine-learning-based text classification. It does not independently verify claims against trusted sources, and its predictions may be incorrect.
