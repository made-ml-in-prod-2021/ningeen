import os
import os.path
import pickle

import click
import pandas as pd
from omegaconf import OmegaConf

from entities.trainer_params import read_trainer_params, TrainerParams
from trainer import train_model, Classifier

TRAINER_CONFIG_PATH = "configs/trainer_config.yaml"


def load_trainer_config() -> TrainerParams:
    omega_data_config = OmegaConf.load(TRAINER_CONFIG_PATH)
    data_config = read_trainer_params(omega_data_config)
    return data_config


def read_data(dataset_path: str, target_path: str) -> (pd.DataFrame, pd.Series):
    df = pd.read_csv(dataset_path)
    target = pd.read_csv(target_path)
    return df, target


def create_dir_if_not_exists(path: str) -> None:
    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def get_out_path(path_from: str, model_file_name: str) -> str:
    path_from, file_name = os.path.split(path_from)
    path_from, date_name = os.path.split(path_from)
    path_from, _ = os.path.split(path_from)
    path_from, _ = os.path.split(path_from)

    path_to = os.path.join(path_from, "models", date_name, model_file_name)
    return path_to


def save_model(path: str, obj: Classifier):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


@click.command("run_train")
@click.argument("model_path")
@click.argument("dataset_in_path")
@click.argument("target_in_path")
def run_train(model_path:str, dataset_in_path: str, target_in_path: str):
    cfg = load_trainer_config()
    df, target = read_data(dataset_in_path, target_in_path)
    clf = train_model(df, target, cfg)
    create_dir_if_not_exists(model_path)
    save_model(model_path, clf)


if __name__ == "__main__":
    run_train()
