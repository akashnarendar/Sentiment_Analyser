from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# Load model once at startup
sentiment_pipe = pipeline("sentiment-analysis", model="models/sentiment-sst2-model")

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(query: Query):
    result = sentiment_pipe(query.text)[0]
    return {"label": result["label"], "confidence": result["score"]}
