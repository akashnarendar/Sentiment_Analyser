from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import numpy as np
import evaluate
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.transformers import log_model
from mlflow import MlflowClient
import pandas as pd
from transformers import pipeline


# MLflow server location
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("Sentiment Analysis")

client = MlflowClient()

checkpoint = "distilbert-base-uncased"
model_name = "sentiment-analyser"
batch_size = 8
num_epochs = 2

# Dataset
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def preprocess(example):
    return tokenizer(example["sentence"], truncation=True)

tokenized_train = dataset["train"].map(preprocess, batched=True)
tokenized_val = dataset["validation"].map(preprocess, batched=True)

# 🔹 Reduce size for faster training
small_train = tokenized_train.shuffle(seed=42).select(range(1000))
small_val = tokenized_val.shuffle(seed=42).select(range(200))

# Model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.config.id2label = {0: "negative", 1: "positive"}
model.config.label2id = {"negative": 0, "positive": 1}

# Metrics
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

# Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# Train and log
with mlflow.start_run() as run:
    trainer.train()
    metrics = trainer.evaluate()

    mlflow.log_params({
        "checkpoint": checkpoint,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "train_size": 1000,
        "val_size": 200,
    })
    mlflow.log_metrics(metrics)

    input_example = {"text": ["This is great!"]}
    signature = infer_signature(pd.DataFrame(input_example), ["positive"])

    # Create a pipeline for inference
    inference_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)


    # Log the model using MLflow
    log_model(
        transformers_model=inference_pipeline,
        artifact_path="model",
        input_example=input_example,
        signature=signature,
    )

    model_uri = f"runs:/{run.info.run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    client.set_registered_model_alias(
        name=model_name,
        alias="prod",
        version=mv.version
    )

    print(f"✅ Model registered as {model_name}@prod (v{mv.version})")
