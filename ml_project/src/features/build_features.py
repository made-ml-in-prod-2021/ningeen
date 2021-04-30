import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from rank_transformer import RankTransformer
from src.entities import FeatureParams


def process_categorical_features(categorical_df: pd.DataFrame) -> pd.DataFrame:
    categorical_pipeline = build_categorical_pipeline()
    processed_df = categorical_pipeline.fit_transform(categorical_df)
    return pd.DataFrame(processed_df.toarray())


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_common")),
            ("ohe", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def process_numerical_features(numerical_df: pd.DataFrame) -> pd.DataFrame:
    numerical_pipeline = build_numerical_pipeline()
    processed_df = numerical_pipeline.fit_transform(numerical_df)
    return pd.DataFrame(processed_df.toarray())


def build_numerical_pipeline():
    numerical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ]
    )
    return numerical_pipeline


def make_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(transformer.transform(df).toarray())


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            ("categorical_pipeline", build_categorical_pipeline(), params.categorical_features),
            ("numerical_pipeline", build_numerical_pipeline(), params.numerical_features),
        ]
    )
    return transformer


def build_rank_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            ("rank_pipeline", RankTransformer(params.method, params.ascending), params.rank_features),
        ]
    )
    return transformer


def build_pipeline(params: FeatureParams) -> Pipeline:
    pipeline = Pipeline(
        [
            ("columns_transformer", build_transformer(params)),
            ("rank_transformer", build_rank_transformer(params)),
        ]
    )
    return pipeline


def extract_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]
    return target
