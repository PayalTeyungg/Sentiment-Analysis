import streamlit as st
import joblib

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Streamlit app
st.title("Sentiment Analysis App")
user_input = st.text_area("Enter your text:")

if st.button("Analyze Sentiment"):
    # Preprocess and predict
    input_vectorized = vectorizer.transform([user_input])
    prediction = model.predict(input_vectorized)[0]
    st.write(f"Predicted Sentiment: {prediction}")