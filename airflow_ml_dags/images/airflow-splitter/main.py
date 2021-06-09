import os
import os.path

import click
import pandas as pd
from omegaconf import OmegaConf

from entities.splitter_params import SplitterParams, read_splitter_params
from splitter import split_data

SPLITTER_CONFIG_PATH = "configs/splitter_config.yaml"


def load_splitter_config() -> SplitterParams:
    omega_splitter_config = OmegaConf.load(SPLITTER_CONFIG_PATH)
    data_config = read_splitter_params(omega_splitter_config)
    return data_config


def read_data(dataset_path: str, target_path: str) -> (pd.DataFrame, pd.Series):
    df = pd.read_csv(dataset_path)
    target = pd.read_csv(target_path)
    return df, target


def create_dir_if_not_exists(path: str) -> None:
    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def get_output_paths(path_from: str) -> (str, str):
    path_from, file_name = os.path.split(path_from)
    path_from, date_name = os.path.split(path_from)
    path_from, _ = os.path.split(path_from)

    file_name, _ = file_name.split(".")

    path_to_train = os.path.join(
        path_from, "splitted", date_name, f"{file_name}_train.csv"
    )
    path_to_test = os.path.join(
        path_from, "splitted", date_name, f"{file_name}_test.csv"
    )
    create_dir_if_not_exists(path_to_train)
    return path_to_train, path_to_test


def save_splitted(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    dataset_in_path: str,
    target_in_path: str,
) -> None:
    dataset_out_path_train, dataset_out_path_test = get_output_paths(dataset_in_path)
    target_out_path_train, target_out_path_test = get_output_paths(target_in_path)

    X_train.to_csv(dataset_out_path_train, index=False)
    X_test.to_csv(dataset_out_path_test, index=False)
    y_train.to_csv(target_out_path_train, index=False)
    y_test.to_csv(target_out_path_test, index=False)


@click.command("run_splitter")
@click.argument("dataset_in_path")
@click.argument("target_in_path")
def run_splitter(dataset_in_path: str, target_in_path: str):
    splitter_config = load_splitter_config()
    df, target = read_data(dataset_in_path, target_in_path)
    X_train, X_test, y_train, y_test = split_data(df, target, splitter_config)
    save_splitted(X_train, X_test, y_train, y_test, dataset_in_path, target_in_path)


if __name__ == "__main__":
    run_splitter()
