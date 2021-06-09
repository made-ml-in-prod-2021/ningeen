import pickle
import os

import click
import pandas as pd

from predictor import predict_proba, Classifier


def read_data(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    return df


def load_model(path: str) -> Classifier:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def create_dir_if_not_exists(path: str) -> None:
    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def save_prediction(path: str, y_pred: pd.Series):
    y_pred.to_csv(path, index=False)


@click.command("run_predict")
@click.argument("model_path")
@click.argument("dataset_in_path")
@click.argument("prediction_path")
def run_predict(model_path: str, dataset_in_path: str, prediction_path: str):
    df = read_data(dataset_in_path)
    clf = load_model(model_path)
    y_pred = predict_proba(df, clf)
    create_dir_if_not_exists(prediction_path)
    save_prediction(prediction_path, y_pred)


if __name__ == "__main__":
    run_predict()
