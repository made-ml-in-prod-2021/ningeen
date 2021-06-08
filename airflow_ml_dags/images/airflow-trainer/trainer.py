import pandas as pd

from entities.trainer_params import read_data_params, PipelineParams
from omegaconf import OmegaConf
import os
import click
from preprocess import preprocess_data

TRAINER_CONFIG_PATH = "./configs/trainer_config.yaml"


def load_pipeline_config() -> PipelineParams:
    omega_data_config = OmegaConf.load(TRAINER_CONFIG_PATH)
    data_config = read_data_params(omega_data_config)
    return data_config


def read_data(dataset_path: str, target_path: str) -> (pd.DataFrame, pd.Series):
    df = pd.read_csv(dataset_path)
    target = pd.read_csv(target_path)
    return df, target


def get_out_path(path_from: str) -> str:
    basename = os.path.basename(path_from)
    dir_path = os.path.dirname(os.path.dirname(path_from))
    path_to = os.path.join(dir_path, "processed", basename)
    return path_to


def save_preprocessed(df: pd.DataFrame, target: pd.Series, dataset_in_path: str, target_in_path: str) -> None:
    dataset_out_path = get_out_path(dataset_in_path)
    target_out_path = get_out_path(target_in_path)

    df.to_csv(dataset_out_path, index=False)
    target.to_csv(target_out_path, index=False)


@click.command("download")
@click.argument("dataset_in_path")
@click.argument("target_in_path")
def run_pipeline(dataset_in_path: str, target_in_path: str):
    pipeline_config = load_pipeline_config()
    df, target = read_data(dataset_in_path, target_in_path)
    df = preprocess_data(df, pipeline_config)

    dataset_out_path = get_out_path(dataset_in_path)
    target_out_path = get_out_path(target_in_path)

    df.to_csv(dataset_out_path, index=False)
    target.to_csv(target_out_path, index=False)


@click.command("download")
@click.argument("dataset_path")
@click.argument("target_path")
def load_new_data(dataset_path: str, target_path: str) -> None:
    data_config = load_data_config()
    dataset, target = get_new_data(data_config)
    create_dir_if_not_exists(dataset_path)
    dataset.to_csv(dataset_path, index=False)
    target.to_csv(target_path, index=False)