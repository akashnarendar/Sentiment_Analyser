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
import os
import sys

# Ensure relative imports work when run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Optional: import evaluation helpers if you create scripts/evaluation.py
# import evaluation

# Set up MLflow tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Sentiment Analysis")

# Config
checkpoint = "distilbert-base-uncased"
num_epochs = 2
batch_size = 8
model_dir = "models/sentiment-sst2-model"

# Load dataset and tokenizer
dataset = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def preprocess(example):
    return tokenizer(example["sentence"], truncation=True)

tokenized_train = dataset["train"].map(preprocess, batched=True)
tokenized_val = dataset["validation"].map(preprocess, batched=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
model.config.id2label = {0: "negative", 1: "positive"}
model.config.label2id = {"negative": 0, "positive": 1}

# Evaluation metric
accuracy_metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)

# Training arguments
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
    report_to=[],  # Disable default reporting
)

small_train = tokenized_train.shuffle(seed=42).select(range(5000))
small_val = tokenized_val.shuffle(seed=42).select(range(500))

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# Start MLflow tracking
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("checkpoint", checkpoint)
    mlflow.log_param("epochs", num_epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("dataset", "glue/sst2")

    # Train the model
    trainer.train()

    # Evaluate and log metrics
    metrics = trainer.evaluate()
    mlflow.log_metrics({
        "eval_accuracy": metrics.get("eval_accuracy", 0),
        "eval_loss": metrics.get("eval_loss", 0)
    })

    # Save model + tokenizer
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    mlflow.log_artifacts(model_dir)
