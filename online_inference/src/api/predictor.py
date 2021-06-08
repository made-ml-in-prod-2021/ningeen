import logging.config
import sys

import numpy as np
import pandas as pd

from src.entities import PredictionPipelineParams, read_prediction_pipeline_params
from src.features import make_features, load_transformer
from src.models import (
    predict_model,
    load_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


class Predictor:
    def __init__(self, params: PredictionPipelineParams):
        self.pipeline = load_transformer(params.transformer_path)
        self.model = load_model(params.model_path)

    @classmethod
    def create_predictor(cls, path):
        params = read_prediction_pipeline_params(path)
        return cls(params)

    def predict_pipeline(self, data: pd.DataFrame) -> np.ndarray:
        logger.info(f"Start prediction.")

        train_features = make_features(self.pipeline, data)
        logger.info(f"Test features shape: {train_features.shape}")

        predictions = predict_model(train_features, self.model)
        logger.info(f"Prediction done")
        return predictions
