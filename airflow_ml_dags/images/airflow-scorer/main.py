import pickle
import json
import os

import click
import pandas as pd
from omegaconf import OmegaConf

from entities.scorer_params import ScorerParams, read_scorer_params
from scorer import score_model, Classifier

SCORER_CONFIG_PATH = "configs/scorer_config.yaml"


def load_scorer_config() -> ScorerParams:
    omega_config = OmegaConf.load(SCORER_CONFIG_PATH)
    scorer_config = read_scorer_params(omega_config)
    return scorer_config


def read_data(target_path: str, prediction_path: str) -> (pd.Series, pd.Series):
    target = pd.read_csv(target_path)
    prediction = pd.read_csv(prediction_path)
    return target, prediction


def load_model(path: str) -> Classifier:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def get_score_path(path_from: str) -> str:
    path_from, file_name = os.path.split(path_from)
    path_to = os.path.join(path_from, "score.json")
    return path_to


def save_score(path: str, obj: dict):
    with open(path, 'w') as f:
        json.dump(obj, f)


@click.command("run_scoring")
@click.argument("target_path")
@click.argument("prediction_path")
def run_scoring(target_path: str, prediction_path: str):
    cfg = load_scorer_config()
    y_true, y_pred = read_data(target_path, prediction_path)
    score = score_model(y_true, y_pred, cfg)
    score_path = get_score_path(prediction_path)
    save_score(score_path, score)


if __name__ == "__main__":
    run_scoring()
