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


@pytest.mark.parametrize(
    "model_type",
    [
        pytest.param("RandomForestClassifier", id='RandomForestClassifier'),
        pytest.param("LogisticRegression", id='LogisticRegression'),
    ]
)
def test_train_e2e(
    tmpdir: LocalPath,
    fake_dataset: str,
    categorical_features: List[str],
    numerical_features: List[str],
    target_col: str,
    features_to_drop: List[str],
    model_type: str
):
    expected_output_model_path = tmpdir.join("model.pkl")
    expected_metric_path = tmpdir.join("metrics.json")
    params = TrainingPipelineParams(
        input_data_path=fake_dataset,
        output_model_path=expected_output_model_path,
        metric_path=expected_metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=239),
        feature_params=FeatureParams(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            target_col=target_col,
            features_to_drop=features_to_drop,
        ),
        train_params=TrainingParams(model_type=model_type),
    )
    real_model_path, metrics = train_pipeline(params)
    assert metrics["accuracy"] >= 0.5
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)
