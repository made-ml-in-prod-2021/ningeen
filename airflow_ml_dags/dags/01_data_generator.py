import os

import airflow
from airflow.models import Variable
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

volume_data = f"{os.environ['VOLUME_PATH']}/data:/data"
default_args = {
    'owner': 'ningeen',
    'start_date': airflow.utils.dates.days_ago(0, hour=3),
    'email': [Variable.get("gmail_user")],
    'email_on_failure': True,
}

with DAG(
    dag_id="01_data_generator",
    schedule_interval="@hourly",
    default_args=default_args,
) as dag:
    get_data = DockerOperator(
        image="airflow-generator",
        command="/data/raw/{{ ds }}/data.csv /data/raw/{{ ds }}/target.csv",
        network_mode="bridge",
        task_id="docker-airflow-generator",
        do_xcom_push=False,
        volumes=[volume_data]
    )
