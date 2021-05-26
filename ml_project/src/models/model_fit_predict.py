import pickle
from typing import Dict, Union
import logging

import numpy as np
import pandas as pd
from entities import TrainingParams
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

SklearnClassifierModel = Union[RandomForestClassifier, LogisticRegression]

logger = logging.getLogger(__name__)


def get_model(params: TrainingParams) -> SklearnClassifierModel:
    if params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(**params.clf_params)
    elif params.model_type == "LogisticRegression":
        model = LogisticRegression(**params.clf_params)
    else:
        logger.error(f"Model type '{params.model_type}' not implemented")
        raise NotImplementedError
    logger.debug(f"Model {params.model_type} initialized")
    return model


def train_model(
    features: pd.DataFrame, target: pd.Series, model: SklearnClassifierModel
) -> SklearnClassifierModel:
    model.fit(features, target)
    logger.debug(f"Model trained")
    return model


def predict_model(features: pd.DataFrame, model: SklearnClassifierModel) -> np.ndarray:
    predictions = model.predict(features)
    logger.debug(f"Predict done")
    return predictions


def evaluate_model(predictions: np.ndarray, target: pd.Series) -> Dict[str, float]:
    metrics = {
        "f1_score": f1_score(target, predictions),
        "accuracy": accuracy_score(target, predictions),
    }
    logger.debug(f"Metrics evaluated")
    return metrics


def serialize_model(model: SklearnClassifierModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    logger.debug(f"Model saved to {output}")
    return output


def load_model(model_path: str) -> SklearnClassifierModel:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.debug(f"Model loaded from {model_path}")
    return model
