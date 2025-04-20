# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc

app = FastAPI()

class SentimentRequest(BaseModel):
    text: str

# ✅ Connect to MLflow tracking server
mlflow.set_tracking_uri("http://mlflow:5000")

# ✅ Load latest production model using alias
model = mlflow.pyfunc.load_model("models:/sentiment-analyser@prod")

@app.post("/predict")
def predict(request: SentimentRequest):
    prediction = model.predict({"text": [request.text]})

    # ✅ Return full prediction (label + score)
    return prediction.iloc[0].to_dict()
