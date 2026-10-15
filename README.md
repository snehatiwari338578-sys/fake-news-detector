# 📰 Fake News Detector

An end-to-end Machine Learning web app that classifies news article text as **Fake** or **Real** using TF-IDF features and Logistic Regression. The app is built with Python and Streamlit.

## 🚀 Live Demo

**Try the app:** https://fake-news-detector-cpwuhaq3jgxv2nnvzrkrwl.streamlit.app/

Paste a news article into the app to see the model's classification and estimated class probabilities.

## ✨ Features

- Classifies article text as **FAKE** or **REAL**
- Converts text into numerical features using TF-IDF
- Uses a trained Logistic Regression model for predictions
- Displays estimated fake/real class probabilities
- Provides an interactive Streamlit interface

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **NLP / Feature extraction:** TF-IDF Vectorizer
- **Models compared:** Logistic Regression, Multinomial Naive Bayes
- **Web app:** Streamlit
- **Deployment:** Streamlit Community Cloud

## 🧠 Workflow

1. Load the Fake and Real news datasets.
2. Assign labels (`0` = Fake, `1` = Real) and combine the datasets.
3. Remove rows with missing or empty article text.
4. Split the data into training and test sets using stratification.
5. Fit TF-IDF on the training text and transform the test text.
6. Train and evaluate Logistic Regression and Multinomial Naive Bayes.
7. Save the trained model and vectorizer, then load them in the Streamlit app.

## 📊 Model Results

| Model | Test accuracy |
|---|---:|
| Logistic Regression | 98.62% |
| Multinomial Naive Bayes | 93.13% |

These results are from the project's held-out test split. They should not be interpreted as real-world fact-checking accuracy; dataset patterns, source clues, or similar articles may affect performance.

## 📂 Project Files

```text
fake-news-detector/
├── app.py                     # Streamlit application
├── fake_news_model.pkl        # Trained Logistic Regression model
├── tfidf_vectorizer.pkl       # Fitted TF-IDF vectorizer
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 💻 Run Locally

1. Clone this repository and open its folder:

   ```bash
   git clone <your-github-repository-url>
   cd fake-news-detector
   ```

2. (Optional) Create and activate a virtual environment.

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

Keep `fake_news_model.pkl` and `tfidf_vectorizer.pkl` in the same folder as `app.py`.

## ⚠️ Limitations

This project performs text classification; it does **not** independently verify claims against trusted news sources or fact-checking databases. A prediction and its probability are model estimates, and the model can be confidently wrong—especially for short statements, unfamiliar topics, or text unlike its training data. Please verify important claims using reliable independent sources.

## 👩‍💻 Author

Add your name and links here:

- **GitHub:** <your-github-profile-url>
- **LinkedIn:** <your-linkedin-profile-url>
