import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from entities import FeatureParams
from .rank_transformer import RankTransformer


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    categorical_pipeline = build_categorical_pipeline()
    processed_df = categorical_pipeline.fit_transform(categorical_df)
    return pd.DataFrame(processed_df.toarray())


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    numerical_pipeline = build_numerical_pipeline()
    processed_df = numerical_pipeline.fit_transform(numerical_df)
    return pd.DataFrame(processed_df.toarray())


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
    return numerical_pipeline


def build_rank_pipeline(params: FeatureParams):
    numerical_pipeline = Pipeline(
        [
            ("rank_pipeline", RankTransformer(params.method, params.ascending)),
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
        ]
    )
    return numerical_pipeline


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    df = transformer.transform(df)
    return pd.DataFrame(df)


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            ("categorical_pipeline", build_categorical_pipeline(), params.categorical_features),
            ("numerical_pipeline", build_numerical_pipeline(params), params.numerical_features),
            ("rank_pipeline", build_rank_pipeline(params), params.rank_features),
        ]
    )
    return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target
