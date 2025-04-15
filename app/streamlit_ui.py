import streamlit as st
from transformers import pipeline

pipe = pipeline("sentiment-analysis", model="models/sentiment-sst2-model")

st.title("Sentiment Analysis")
text = st.text_input("Enter a sentence:")

if st.button("Analyze"):
    result = pipe(text)[0]
    st.write(f"**Prediction:** {result['label']} with {round(result['score'] * 100, 2)}% confidence")
