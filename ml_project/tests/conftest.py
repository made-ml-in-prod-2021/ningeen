import os
import random

import numpy as np
import pandas as pd
import pytest
from faker import Faker
from typing import List, Dict, Tuple, DefaultDict, Union
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class TestConfig:
    test_fname: str
    target_col: str
    threshold_col: str
    splitting_val_size: float
    min_accuracy: float
    splitting_random_state: int
    target_threshold: int
    categorical_features_size: int
    numerical_features_size: int
    features_to_drop_size: int
    column_names: List[str]
    possible_cat_feat: List[str]
    limits: Dict[str, List[int]]
    rank_params: Dict[str, List[Union[str, bool]]]
    model_types: List[str]
    dataset_size: int = field(default=100)
    proba: float = field(default=0.05)


TEST_CONFIG_PATH = "./configs/test_config.yaml"
TestSchema = class_schema(TestConfig)


def load_test(test_path=TEST_CONFIG_PATH):
    with open(test_path) as file:
        config_test = yaml.safe_load(file)
        schema = TestSchema()
        return schema.load(config_test)


config_test = load_test()


@pytest.fixture(scope="session")
def config_test_fixture():
    return config_test


@pytest.fixture(scope="session")
def dataset_size():
    return config_test.dataset_size


@pytest.fixture(scope="session")
def target_col():
    return config_test.target_col


@pytest.fixture
def categorical_features() -> List[str]:
    return np.random.choice(
        config_test.possible_cat_feat,
        size=config_test.categorical_features_size,
        replace=False,
    ).tolist()


@pytest.fixture
def numerical_features() -> List[str]:
    return np.random.choice(
        config_test.column_names,
        size=config_test.numerical_features_size,
        replace=False,
    ).tolist()


@pytest.fixture
def features_to_drop() -> List[str]:
    return np.random.choice(
        config_test.column_names,
        size=config_test.features_to_drop_size,
        replace=False,
    ).tolist()


@pytest.fixture(scope="function")
def fake_dataset(tmpdir):
    dataset_path = os.path.join(tmpdir, config_test.test_fname)
    faker = Faker()

    dataset = {}
    for col in config_test.column_names:
        items = []
        for _ in range(config_test.dataset_size):
            value = None
            if random.random() > config_test.proba:
                value = faker.pyint(*config_test.limits[col])
            items.append(value)
        dataset[col] = items

    dataset = pd.DataFrame(dataset)
    dataset["target"] = (
        dataset[config_test.threshold_col] > config_test.target_threshold
    ).astype(int)
    dataset.to_csv(dataset_path, index=False)
    return dataset_path
