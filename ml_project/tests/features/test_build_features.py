import pytest
import random
import pandas as pd
from numpy.testing import assert_allclose

from src.data.make_dataset import read_data
from src.entities.feature_params import FeatureParams
from src.features.build_features import make_features, extract_target, build_transformer


def test_make_features(
    fake_dataset: str,
    target_col: str,
    config_test_fixture,
):
    def generate_cols_params():
        column_names = config_test_fixture.column_names.copy()
        random.shuffle(column_names)
        indices = [random.randint(0, len(column_names)) for _ in range(3)]
        indices.sort()
        param = (
            column_names[slice(0, indices[0])],
            column_names[slice(indices[0], indices[1])],
            column_names[slice(indices[1], indices[2])],
            column_names[slice(indices[2], len(column_names))],
            bool(random.randint(0, 1)),
        )
        return param

    data = read_data(fake_dataset)

    for _ in range(10):
        categorical_features, numerical_features, rank_features, features_to_drop, normalize = generate_cols_params()
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
