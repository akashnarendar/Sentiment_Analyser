from transformers import pipeline

# Load the trained model from saved path
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="models/sentiment-sst2-model",
    tokenizer="models/sentiment-sst2-model"
)

# Test predictions
print(sentiment_pipe("I'm thrilled with how this project turned out!"))
print(sentiment_pipe("This was the worst experience ever."))
