import os
from typing import List
import pytest

from py._path.local import LocalPath

from src.train_pipeline import train_pipeline
from src.entities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
)


def test_train_e2e(
    tmpdir: LocalPath,
    fake_dataset: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
    features_to_drop: List[str],
    config_test
):
    categorical_features = list(set(categorical_features) - set(numerical_features))
    features_to_drop = list(set(features_to_drop) - set(numerical_features))
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    expected_transformer_path = tmpdir.join("transformer.pkl")
    for model_type in config_test.model_types:
        params = TrainingPipelineParams(
            input_data_path=fake_dataset,
            input_data_url="",
            output_model_path=expected_output_model_path,
            metric_path=expected_metric_path,
            transformer_path=expected_transformer_path,
            splitting_params=SplittingParams(
                val_size=config_test.splitting_val_size,
                random_state=config_test.splitting_random_state,
            ),
            feature_params=FeatureParams(
                numerical_features=numerical_features,
                categorical_features=categorical_features,
                target_col=target_col,
                features_to_drop=features_to_drop,
            ),
            train_params=TrainingParams(model_type=model_type),
        )
        real_model_path, metrics = train_pipeline(params)
        assert metrics["accuracy"] >= config_test.min_accuracy
        assert os.path.exists(real_model_path)
        assert os.path.exists(params.metric_path)
