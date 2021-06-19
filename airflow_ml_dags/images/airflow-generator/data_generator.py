import os.path
import random

import click
import pandas as pd
from faker import Faker
from omegaconf import OmegaConf

from entities.data_params import read_data_params, DataParams

DATA_CONFIG_PATH = "./configs/load_data_config.yaml"


def load_data_config() -> DataParams:
    omega_data_config = OmegaConf.load(DATA_CONFIG_PATH)
    data_config = read_data_params(omega_data_config)
    return data_config


def get_new_data(data_config: DataParams) -> (pd.DataFrame, pd.Series):
    dataset = generate_fake_dataset(data_config)

    dataset = pd.DataFrame(dataset)
    target = pd.Series(
        dataset[data_config.threshold_col] > data_config.target_threshold,
        name=data_config.target_col,
        dtype=int,
    )

    return dataset, target


def generate_fake_dataset(data_config: DataParams):
    faker = Faker()
    dataset = {}
    for col in data_config.column_names:
        items = []
        for _ in range(data_config.dataset_size):
            value = None
            if random.random() > data_config.probability:
                value = faker.pyint(*data_config.limits[col])
            items.append(value)
        dataset[col] = items
    return dataset


def create_dir_if_not_exists(path: str) -> None:
    dir_path = os.path.dirname(path)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


@click.command("download")
@click.argument("dataset_path")
@click.argument("target_path")
def load_new_data(dataset_path: str, target_path: str) -> None:
    data_config = load_data_config()
    dataset, target = get_new_data(data_config)
    create_dir_if_not_exists(dataset_path)
    dataset.to_csv(dataset_path, index=False)
    target.to_csv(target_path, index=False)


if __name__ == "__main__":
    load_new_data()
