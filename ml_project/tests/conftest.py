import os
import random

import pandas as pd
import pytest
from faker import Faker
from typing import List

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
    return ["thal", "slope", 'fbs', "cp", "chol", 'oldpeak', "trestbps", 'thalach', "restecg", "ca", 'exang']


@pytest.fixture()
def fake_dataset(tmpdir):
    dataset_path = os.path.join(tmpdir, "test_dataset.csv")
    faker = Faker()
    dataset = pd.DataFrame({
        "age": [faker.pyint(20, 80) if random.random() > PROBA else None for _ in range(DATASET_SIZE)],
        "sex": [faker.pyint(0, 1) if random.random() > PROBA else None for _ in range(DATASET_SIZE)],
        "cp": [faker.pyint(0, 3) if random.random() > PROBA else None for _ in range(DATASET_SIZE)],
        "trestbps": [faker.pyint(90, 200) if random.random() > PROBA else None for _ in range(DATASET_SIZE)],
        "chol": [faker.pyint(120, 570) if random.random() > PROBA else None for _ in range(DATASET_SIZE)],
        "fbs": [faker.pyint(0, 1) if random.random() > PROBA else None for _ in range(DATASET_SIZE)],
        "restecg": [faker.pyint(0, 2) if random.random() > PROBA else None for _ in range(DATASET_SIZE)],
        "thalach": [faker.pyint(70, 210) if random.random() > PROBA else None for _ in range(DATASET_SIZE)],
        "exang": [faker.pyint(0, 1) if random.random() > PROBA else None for _ in range(DATASET_SIZE)],
        "oldpeak": [faker.pyfloat(0, 7) if random.random() > PROBA else None for _ in range(DATASET_SIZE)],
        "slope": [faker.pyint(0, 2) if random.random() > PROBA else None for _ in range(DATASET_SIZE)],
        "ca": [faker.pyint(0, 4) if random.random() > PROBA else None for _ in range(DATASET_SIZE)],
        "thal": [faker.pyint(0, 3) if random.random() > PROBA else None for _ in range(DATASET_SIZE)],
    })
    dataset["target"] = (dataset["age"] > 50).astype(int)
    dataset.to_csv(dataset_path, index=False)
    return dataset_path
