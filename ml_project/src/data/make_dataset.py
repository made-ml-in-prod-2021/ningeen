import os
import urllib.request
from typing import Tuple

import pandas as pd
from entities import SplittingParams
from sklearn.model_selection import train_test_split


def load_data(path: str, url=None):
    if not os.path.isfile(path):
        if url is not None:
            dirname = os.path.dirname(path)
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            urllib.request.urlretrieve(url, path)
        else:
            raise FileNotFoundError


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data
