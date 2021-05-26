import pickle
import logging

import numpy as np
import pandas as pd
from entities import FeatureParams
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .rank_transformer import RankTransformer

logger = logging.getLogger(__name__)


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    logger.debug("Categorical pipeline built")
    return categorical_pipeline


def build_numerical_pipeline(params: FeatureParams):
    if params.normalize:
        numerical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numerical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ]
        )
    logger.debug("Numerical pipeline built")
    return numerical_pipeline


def build_rank_pipeline(params: FeatureParams):
    numerical_pipeline = Pipeline(
        [
            ("rank_pipeline", RankTransformer(params.method, params.ascending)),
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ]
    )
    logger.debug("Rank pipeline built")
    return numerical_pipeline


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    df = transformer.transform(df)
    logger.debug("Features made")
    return pd.DataFrame(df)


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(params),
                params.numerical_features,
            ),
            ("rank_pipeline", build_rank_pipeline(params), params.rank_features),
        ]
    )
    logger.debug("Transformer built")
    return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    logger.debug("Target extracted")
    return target


def save_transformer(transformer: ColumnTransformer, output_path: str) -> str:
    with open(output_path, "wb") as f:
        pickle.dump(transformer, f)
    logger.debug(f"Transformer saved to {output_path}")
    return output_path


def load_transformer(transformer_path: str) -> ColumnTransformer:
    with open(transformer_path, "rb") as f:
        transformer = pickle.load(f)
    logger.debug(f"Transformer loaded from {transformer_path}")
    return transformer
