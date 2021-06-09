import pytest
import os
from airflow.models import DagBag

os.environ['VOLUME_PATH'] = "/home/ningeen/Documents/repos/ml_in_prod/airflow_ml_dags"


@pytest.fixture
def dag_bag():
    dag = DagBag(dag_folder="dags/", include_examples=False)
    return dag


def test_dag_bag_import(dag_bag):
    assert dag_bag.dags is not None
    assert "01_data_generator" in dag_bag.dags
    assert "02_pipeline" in dag_bag.dags
    assert "03_prediction" in dag_bag.dags


@pytest.mark.parametrize(
    "dag_id, num_tasks",
    [
        pytest.param("01_data_generator", 1),
        pytest.param("02_pipeline", 7),
        pytest.param("03_prediction", 4),
    ],
)
def test_dag_loaded(dag_bag, dag_id, num_tasks):
    dag = dag_bag.dags[dag_id]
    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == num_tasks


def test_data_generator_dag(dag_bag):
    flow = {
        "docker-airflow-generator": set(),
    }
    dag = dag_bag.dags["01_data_generator"]
    for name, task in dag.task_dict.items():
        assert flow[name] == task.downstream_task_ids


def test_pipeline_dag(dag_bag):
    flow = {
        "wait_raw_data": {'docker-airflow-preprocessor'},
        "wait_raw_target": {'docker-airflow-preprocessor'},
        "docker-airflow-preprocessor": {'docker-airflow-splitter'},
        "docker-airflow-splitter": {'docker-airflow-trainer'},
        "docker-airflow-trainer": {'docker-airflow-predict'},
        "docker-airflow-predict": {'docker-airflow-scorer'},
        "docker-airflow-scorer": set(),
    }
    dag = dag_bag.dags["02_pipeline"]
    for name, task in dag.task_dict.items():
        assert flow[name] == task.downstream_task_ids


def test_prediction_dag(dag_bag):
    flow = {
        "wait_raw_data": {'docker-airflow-preprocessor'},
        "wait_raw_target": {'docker-airflow-preprocessor'},
        "docker-airflow-preprocessor": {'docker-airflow-predict'},
        "docker-airflow-predict": set(),
    }
    dag = dag_bag.dags["03_prediction"]
    for name, task in dag.task_dict.items():
        assert flow[name] == task.downstream_task_ids
