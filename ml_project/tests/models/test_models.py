import os
import pickle
from typing import List, Tuple, Union

import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data.make_dataset import read_data
from src.entities.feature_params import FeatureParams
from src.features.build_features import make_features, extract_target, build_transformer
from src.models.model_fit_predict import train_model, serialize_model

SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]


def create_object_by_type(obj_type):
    return globals()[obj_type]


@pytest.fixture()
def features_and_target(
    fake_dataset: str,
    target_col: str,
    categorical_features: List[str],
    numerical_features: List[str],
    features_to_drop: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
    )
    data = read_data(fake_dataset)
    transformer = build_transformer(params)
    transformer.fit(data)
    features = make_features(transformer, data)
    target = extract_target(data, params)
    return features, target


def test_train_model(
    config_test_fixture, features_and_target: Tuple[pd.DataFrame, pd.Series]
):
    for clf in config_test_fixture.model_types:
        clf = create_object_by_type(clf)
        features, target = features_and_target
        model = train_model(features, target, clf())
        assert isinstance(model, clf)
        assert model.predict(features).shape[0] == target.shape[0]


def test_serialize_model(tmpdir: LocalPath, config_test_fixture):
    for clf in config_test_fixture.model_types:
        clf = create_object_by_type(clf)
        expected_output = tmpdir.join("model.pkl")
        model = clf()
        real_output = serialize_model(model, expected_output)
        assert real_output == expected_output
        assert os.path.exists
        with open(real_output, "rb") as f:
            model = pickle.load(f)
        assert isinstance(model, clf)
