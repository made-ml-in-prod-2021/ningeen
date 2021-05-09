import os
from typing import List

import pytest

from src.data import generate_dataset

DATASET_SIZE = 100
PROBA = 0.05


@pytest.fixture
def dataset_size():
    return DATASET_SIZE


@pytest.fixture
def target_col():
    return "target"


@pytest.fixture
def categorical_features() -> List[str]:
    return ["sex"]


@pytest.fixture
def numerical_features() -> List[str]:
    return ["age"]


@pytest.fixture
def features_to_drop() -> List[str]:
    return [
        "thal",
        "slope",
        "fbs",
        "cp",
        "chol",
        "oldpeak",
        "trestbps",
        "thalach",
        "restecg",
        "ca",
        "exang",
    ]


@pytest.fixture()
def fake_dataset(tmpdir):
    dataset_path = os.path.join(tmpdir, "test_dataset.csv")
    dataset = generate_dataset(DATASET_SIZE, PROBA)
    dataset.to_csv(dataset_path, index=False)
    return dataset_path
