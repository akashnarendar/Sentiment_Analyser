from mlflow import MlflowClient
from mlflow.exceptions import RestException
import subprocess
import sys

MODEL_NAME = "sentiment-analyser"
ALIAS_NAME = "prod"
MLFLOW_TRACKING_URI = "http://mlflow:5000"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

try:
    # 🔍 Check if model exists first
    registered_models = [rm.name for rm in client.search_registered_models()]
    if MODEL_NAME not in registered_models:
        raise RestException({})  # Fake trigger to run training below

    # ✅ Now check if alias exists
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS_NAME)
    print(f"✅ Found model with alias '{ALIAS_NAME}' for '{MODEL_NAME}' (version {mv.version})")
    print("➡️ Skipping training since model already exists.")
    sys.exit(0)

except RestException:
    print(f"⚠️ Model or alias not found. Triggering training...")
    subprocess.run(["python", "train.py"], check=True)
