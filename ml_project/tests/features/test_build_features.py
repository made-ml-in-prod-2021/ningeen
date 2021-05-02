import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.data.make_dataset import read_data
from src.entities.feature_params import FeatureParams
from src.features.build_features import make_features, extract_target, build_transformer


@pytest.mark.parametrize(
    "categorical_features, numerical_features, rank_features, features_to_drop, normalize",
    [
        pytest.param(
            ["sex", "cp", "chol", "oldpeak"],
            ["age", "trestbps", "thalach"],
            ["restecg", "ca", "exang"],
            ["thal", "slope", "fbs"],
            True,
            id="0",
        ),
        pytest.param(
            [],
            ["age", "trestbps", "thalach", "sex", "cp", "chol", "oldpeak"],
            ["restecg", "ca", "exang"],
            ["thal", "slope", "fbs"],
            False,
            id="1",
        ),
        pytest.param(
            ["age", "trestbps", "thalach", "sex", "cp", "chol", "oldpeak"],
            [],
            ["restecg", "ca", "exang"],
            ["thal", "slope", "fbs"],
            False,
            id="2",
        ),
        pytest.param(
            ["sex"],
            ["age"],
            [],
            [
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
            ],
            True,
            id="3",
        ),
        pytest.param(
            ["sex"],
            ["age"],
            [
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
            ],
            [],
            True,
            id="4",
        ),
    ],
)
def test_make_features(
    fake_dataset: str,
    target_col: str,
    categorical_features: list,
    numerical_features: list,
    rank_features: list,
    features_to_drop: list,
    normalize: bool,
):
    data = read_data(fake_dataset)
    feature_params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        rank_features=rank_features,
        features_to_drop=features_to_drop,
        normalize=normalize,
        target_col=target_col,
    )
    transformer = build_transformer(feature_params)
    transformer.fit(data)
    features = make_features(transformer, data)
    assert not pd.isnull(features).any().any()
    assert all(x not in features.columns for x in feature_params.features_to_drop)

    target = extract_target(data, feature_params)
    assert_allclose(data[feature_params.target_col].to_numpy(), target.to_numpy())
