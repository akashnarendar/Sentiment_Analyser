# 🧠 Sentiment Analyser — End-to-End NLP Pipeline

An end-to-end Sentiment Analysis project using DistilBERT, powered by Hugging Face Transformers, tracked with MLflow, orchestrated using Airflow, and served through a REST API (FastAPI) and a user-friendly Streamlit UI — all containerized and ready for cloud deployment 🚀

## 📌 Project Highlights

- ✅ Transformer-based sentiment analysis (DistilBERT fine-tuned on SST2)
- ✅ MLflow tracking for metrics, artifacts, and model lifecycle
- ✅ Airflow orchestration for automated pipeline scheduling
- ✅ Dockerized end-to-end for easy deployment
- ✅ FastAPI backend to expose model as a service
- ✅ Streamlit UI for non-technical users
- ✅ AWS-ready (SageMaker / EC2 compatible)

## 🗂️ Project Structure

sentiment-analyser/
├── app/                      # FastAPI and Streamlit interfaces
│   ├── main.py               # FastAPI API service
│   └── streamlit_ui.py       # Streamlit user interface
├── scripts/                  # Training, evaluation, inference
│   ├── train.py
│   ├── inference.py
│   └── evaluation.py
├── airflow/                  # Airflow DAGs
│   └── dags/
│       └── train_pipeline.py
├── Dockerfile                # Docker container for app
├── docker-compose.yml        # MLflow + Airflow + QnA app orchestration
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md

## ⚙️ How to Run

1. Clone the repo

   git clone https://github.com/akashnarendar/sentiment-analyser.git
   cd sentiment-analyser

2. Build and Launch

   docker-compose up --build

This launches:

- 🧠 ML training and logging (Airflow)
- 📊 MLflow dashboard on http://localhost:5050
- 🧪 FastAPI on http://localhost:8000/docs
- 🎨 Streamlit UI on http://localhost:8501

## 🧪 API Inference (FastAPI)

POST /predict

{
  "text": "This product was absolutely amazing!"
}

Response:

{
  "label": "positive",
  "score": 0.9732
}

## 🎨 Streamlit App

A clean UI to test your model for human convenience:

streamlit run app/streamlit_ui.py

## 📊 MLflow Tracking

Track model parameters, accuracy, artifacts, and checkpoints at:

http://localhost:5050

## ⏰ Airflow Scheduler

You can trigger DAGs like train_pipeline to automatically rebuild your model from data:

http://localhost:8080
login: admin / admin

## 🧠 Model Used

- distilbert-base-uncased fine-tuned on GLUE SST-2 (Stanford Sentiment Treebank)
- Lightweight and efficient for real-time inference
- ⚠️ For production-grade accuracy, consider using bert-base-cased, roberta, or DeBERTa.

## 📦 Future Enhancements

- ✅ Model registry with MLflow
- ✅ Dockerized endpoints for AWS
- ☁️ Deploy to SageMaker / Lambda
- 📥 Add ability to upload CSV for bulk sentiment processing
- 📈 Add confidence interval and explainability

## 🙌 Author

Narendar Punithan  
LinkedIn: https://www.linkedin.com/in/narendar-punithan-758658126  
GitHub: https://github.com/akashnarendar

> 🚀 This is part of my upcoming ML Portfolio Series. Stay tuned for more cutting-edge projects!
