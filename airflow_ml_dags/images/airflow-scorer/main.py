import pickle

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


def read_data(dataset_path: str, target_path: str) -> (pd.DataFrame, pd.Series):
    df = pd.read_csv(dataset_path)
    target = pd.read_csv(target_path)
    return df, target


def load_model(path: str) -> Classifier:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


@click.command("run_train")
@click.argument("model_path")
@click.argument("dataset_in_path")
@click.argument("target_in_path")
def run_train(model_path: str, dataset_in_path: str, target_in_path: str):
    cfg = load_scorer_config()
    df, target = read_data(dataset_in_path, target_in_path)
    clf = load_model(model_path)
    score = score_model(df, target, clf, cfg)
    print(score)


if __name__ == "__main__":
    run_train()
