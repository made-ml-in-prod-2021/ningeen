import os

import airflow
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

volume_data = f"{os.environ['VOLUME_PATH']}/data:/data"
volume_models = f"{os.environ['VOLUME_PATH']}/models:/models"

with DAG(
    dag_id="03_prediction",
    schedule_interval="@daily",
    start_date=airflow.utils.dates.days_ago(7),
) as dag:

    wait_raw_data = FileSensor(
        task_id="wait_raw_data",
        filepath="./data/raw/{{ ds }}/data.csv",
        poke_interval=30,
        retries=100,
    )

    wait_raw_target = FileSensor(
        task_id="wait_raw_target",
        filepath="./data/raw/{{ ds }}/target.csv",
        poke_interval=30,
        retries=100,
    )

    preprocessor = DockerOperator(
        image="airflow-preprocessor",
        command="/data/raw/{{ ds }}/data.csv /data/raw/{{ ds }}/target.csv",
        network_mode="bridge",
        task_id="docker-airflow-preprocessor",
        do_xcom_push=False,
        volumes=[volume_data],
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="/models/{{ ds }}/clf.pkl /data/splitted/{{ ds }}/data_test.csv /data/predictions/{{ ds }}/prediction.csv",
        network_mode="bridge",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=[volume_data, volume_models],
    )

    [wait_raw_data, wait_raw_target] >> preprocessor >> predict
