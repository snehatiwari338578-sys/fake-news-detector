# TruthLens — Fake News Detection

TruthLens is a machine learning project that classifies news articles as Fake or Real. I built the classification pipeline using Python and Scikit-learn, connected it to a FastAPI backend, and created the web interface using Lovable.

## Live Demo

- **Frontend:** https://intel-scan.lovable.app/
- **Backend API:** https://fake-news-detector-b7pz.onrender.com
- **API Documentation:** https://fake-news-detector-b7pz.onrender.com/docs

## How It Works

1. A user pastes a news article into the website.
2. The frontend sends the article to the Python backend.
3. The TF-IDF vectorizer converts the text into numerical features.
4. The trained Logistic Regression model predicts the class.
5. FastAPI returns the prediction and estimated probabilities.
6. The frontend displays the result.

## Tech Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn, Logistic Regression, Multinomial Naive Bayes
- **Text Processing:** TF-IDF
- **Backend:** FastAPI, Uvicorn
- **Frontend:** Lovable-generated web interface
- **Deployment:** Render and Lovable

## Dataset and Training

The project uses the Fake and Real News Dataset, with data loaded from `Fake.csv` and `True.csv`.

The training process included:
- Combining the fake and real news datasets.
- Removing missing and empty article text.
- Assigning labels to fake and real articles.
- Splitting the data into training and testing sets using an 80/20 stratified split.
- Converting article text into TF-IDF features using a maximum of 5,000 features.
- Training and comparing two classification models.

## Model Results

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 98.62% |
| Multinomial Naive Bayes | 93.13% |

Logistic Regression performed better on this test split and was used for the deployed prediction API.

These results reflect performance on the selected dataset split, not accuracy at verifying real-world news. Dataset patterns and similar articles can affect the results.

## Run the Backend Locally

Install the dependencies:

`pip install -r requirements.txt`

Start the API from the project directory:

`uvicorn api:app --reload`

Open the API documentation:

`http://127.0.0.1:8000/docs`

The saved model and TF-IDF vectorizer files must be present in the project directory.

## Limitations and Future Improvements

- The model may classify short statements or unfamiliar topics incorrectly.
- It identifies text patterns rather than independently checking claims against trusted sources.
- Future improvements could include duplicate detection, more diverse datasets, stronger evaluation methods, and evidence retrieval from reliable sources.

## What I Learned

This project helped me understand the workflow from preparing text data and training classification models to exposing predictions through an API and connecting a web interface to a deployed backend.

## Author

Add your GitHub and LinkedIn profile links here.
