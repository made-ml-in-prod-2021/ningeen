import os
import os.path

import click
import pandas as pd
from omegaconf import OmegaConf

from entities.preprocess_params import PreprocessParams, read_preprocess_params
from preprocess import preprocess_data

PREPROCESS_CONFIG_PATH = "./configs/preprocess_config.yaml"


def load_preprocess_config() -> PreprocessParams:
    omega_preprocess_config = OmegaConf.load(PREPROCESS_CONFIG_PATH)
    data_config = read_preprocess_params(omega_preprocess_config)
    return data_config


def read_data(dataset_path: str, target_path: str) -> (pd.DataFrame, pd.Series):
    df = pd.read_csv(dataset_path)
    target = pd.read_csv(target_path)
    return df, target


def create_dir_if_not_exists(path: str) -> None:
    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def get_output_path(path_from: str) -> str:
    path_from, file_name = os.path.split(path_from)
    path_from, date_name = os.path.split(path_from)
    path_from, _ = os.path.split(path_from)

    path_to = os.path.join(path_from, "processed", date_name, file_name)
    create_dir_if_not_exists(path_to)
    return path_to


def save_preprocessed(
    df: pd.DataFrame, target: pd.Series, dataset_in_path: str, target_in_path: str
) -> None:
    dataset_out_path = get_output_path(dataset_in_path)
    target_out_path = get_output_path(target_in_path)

    df.to_csv(dataset_out_path, index=False)
    target.to_csv(target_out_path, index=False)


@click.command("run_preprocess")
@click.argument("dataset_in_path")
@click.argument("target_in_path")
def run_preprocess(dataset_in_path: str, target_in_path: str):
    preprocess_config = load_preprocess_config()
    df, target = read_data(dataset_in_path, target_in_path)
    df = preprocess_data(df, preprocess_config)
    save_preprocessed(df, target, dataset_in_path, target_in_path)


if __name__ == "__main__":
    run_preprocess()
