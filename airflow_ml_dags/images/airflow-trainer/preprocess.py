import os.path

import numpy as np
import pandas as pd
from typing import List, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from entities.trainer_params import PreprocessParams


def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df.drop(columns=columns)


def scale_columns(df: pd.DataFrame, columns: List[str], scaler: Union[MinMaxScaler, StandardScaler]) -> pd.DataFrame:
    for col in columns:
        df[col] = scaler.fit_transform(df[col])
    return df


def preprocess_data(df: pd.DataFrame, cfg: PreprocessParams):
    df = drop_columns(df, cfg.cols_to_drop)
    scaler = globals()[cfg.scaler]()
    df = scale_columns(df, cfg.cols_to_scale, scaler)
    return df

