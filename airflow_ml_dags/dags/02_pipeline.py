import airflow
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

with DAG(
    dag_id="02_pipeline",
    schedule_interval="@weekly",
    start_date=airflow.utils.dates.days_ago(7),
) as dag:
    preprocessor = DockerOperator(
        image="airflow-preprocessor",
        command="/data/raw/{{ ds }}/data.csv /data/raw/{{ ds }}/target.csv",
        network_mode="bridge",
        task_id="docker-airflow-preprocessor",
        do_xcom_push=False,
        volumes=["/home/ningeen/Documents/repos/ml_in_prod/airflow_ml_dags/data:/data"],
    )

    splitter = DockerOperator(
        image="airflow-splitter",
        command="/data/processed/{{ ds }}/data.csv /data/processed/{{ ds }}/target.csv",
        network_mode="bridge",
        task_id="docker-airflow-splitter",
        do_xcom_push=False,
        volumes=["/home/ningeen/Documents/repos/ml_in_prod/airflow_ml_dags/data:/data"],
    )

    trainer = DockerOperator(
        image="airflow-trainer",
        command="/models/{{ ds }}/clf.pkl /data/splitted/{{ ds }}/data_train.csv /data/splitted/{{ ds }}/target_train.csv",
        network_mode="bridge",
        task_id="docker-airflow-trainer",
        do_xcom_push=False,
        volumes=[
            "/home/ningeen/Documents/repos/ml_in_prod/airflow_ml_dags/data:/data",
            "/home/ningeen/Documents/repos/ml_in_prod/airflow_ml_dags/models:/models",
        ],
    )

    scorer = DockerOperator(
        image="airflow-scorer",
        command="/models/{{ ds }}/clf.pkl /data/splitted/{{ ds }}/data_test.csv /data/splitted/{{ ds }}/target_test.csv",
        network_mode="bridge",
        task_id="docker-airflow-scorer",
        do_xcom_push=False,
        volumes=[
            "/home/ningeen/Documents/repos/ml_in_prod/airflow_ml_dags/data:/data",
            "/home/ningeen/Documents/repos/ml_in_prod/airflow_ml_dags/models:/models",
        ],
    )

    preprocessor >> splitter >> trainer >> scorer
