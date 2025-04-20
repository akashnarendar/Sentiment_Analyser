from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG("train_sentiment_model", default_args=default_args, schedule_interval=None, catchup=False) as dag:
    train_model = BashOperator(
        task_id="run_training",
        bash_command="docker-compose run --rm trainer"
    )
