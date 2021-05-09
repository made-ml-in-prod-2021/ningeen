import random

import pandas as pd
from faker import Faker


def generate_dataset(size, nan_probability):
    faker = Faker()
    dataset = pd.DataFrame(
        {
            "age": [
                faker.pyint(20, 80) if random.random() > nan_probability else None
                for _ in range(size)
            ],
            "sex": [
                faker.pyint(0, 1) if random.random() > nan_probability else None
                for _ in range(size)
            ],
            "cp": [
                faker.pyint(0, 3) if random.random() > nan_probability else None
                for _ in range(size)
            ],
            "trestbps": [
                faker.pyint(90, 200) if random.random() > nan_probability else None
                for _ in range(size)
            ],
            "chol": [
                faker.pyint(120, 570) if random.random() > nan_probability else None
                for _ in range(size)
            ],
            "fbs": [
                faker.pyint(0, 1) if random.random() > nan_probability else None
                for _ in range(size)
            ],
            "restecg": [
                faker.pyint(0, 2) if random.random() > nan_probability else None
                for _ in range(size)
            ],
            "thalach": [
                faker.pyint(70, 210) if random.random() > nan_probability else None
                for _ in range(size)
            ],
            "exang": [
                faker.pyint(0, 1) if random.random() > nan_probability else None
                for _ in range(size)
            ],
            "oldpeak": [
                faker.pyfloat(0, 7) if random.random() > nan_probability else None
                for _ in range(size)
            ],
            "slope": [
                faker.pyint(0, 2) if random.random() > nan_probability else None
                for _ in range(size)
            ],
            "ca": [
                faker.pyint(0, 4) if random.random() > nan_probability else None
                for _ in range(size)
            ],
            "thal": [
                faker.pyint(0, 3) if random.random() > nan_probability else None
                for _ in range(size)
            ],
        }
    )
    dataset["target"] = (dataset["age"] > 50).astype(int)
    return dataset
