import mlflow
from datetime import datetime
import os

def log_prediction_to_mlflow(input_text: str, prediction: str, log_file="mlflow_logs/predictions.txt"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Compose log message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"[{timestamp}]\n"
        f"Input: {input_text}\n"
        f"Prediction: {prediction}\n"
        f"{'-'*40}\n"
    )

    # Append to local file
    with open(log_file, "a") as f:
        f.write(log_entry)

    # Log to MLflow
    mlflow.log_artifact(log_file)
