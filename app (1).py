
import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Page settings
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Website heading
st.title("📰 Fake News Detector")
st.write(
    "Enter a news article below to classify it using "
    "our Machine Learning model."
)

# Input box
article = st.text_area(
    "Paste your news article here:",
    height=200,
    placeholder="Enter the news article..."
)

# Prediction button
if st.button("Check News"):
    if article.strip() == "":
        st.warning("Please enter a news article first.")

    else:
        article_tfidf = tfidf.transform([article])
        prediction = model.predict(article_tfidf)[0]
        probabilities = model.predict_proba(article_tfidf)[0]

        if prediction == 0:
            st.error("Prediction: FAKE NEWS")
        else:
            st.success("Prediction: REAL NEWS")

        st.write(
            f"Fake probability: {probabilities[0] * 100:.2f}%"
        )
        st.write(
            f"Real probability: {probabilities[1] * 100:.2f}%"
        )

        st.caption(
            "This is an ML classification, not verified fact-checking. "
            "The model can make incorrect predictions."
        )
