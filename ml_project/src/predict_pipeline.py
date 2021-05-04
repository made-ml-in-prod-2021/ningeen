import logging.config

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from data import read_data
from entities import PredictionPipelineParams, read_prediction_pipeline_params
from features import make_features, load_transformer
from models import (
    predict_model,
    load_model,
)
from utils import save_config

logger = logging.getLogger(__name__)


def save_prediction(predictions: np.ndarray, output_path: str) -> str:
    pd.Series(predictions).to_csv(output_path, index=False)
    return output_path


def predict_pipeline(params: PredictionPipelineParams):
    logger.info(f"Start prediction.")

    data = read_data(params.data_path)
    logger.info(f"Data loaded. Raw data shape: {data.shape}")

    pipeline = load_transformer(params.transformer_path)
    logger.info(f"Transformer loaded: {pipeline}")

    model = load_model(params.model_path)
    logger.info(f"Model loaded: {model}")

    train_features = make_features(pipeline, data)
    logger.info(f"Test features shape: {train_features.shape}")

    predictions = predict_model(train_features, model)
    predictions_path = save_prediction(predictions, params.output_path)
    logger.info(f"Predictions saved in {predictions_path}")


@hydra.main(config_path="../configs/", config_name="config_pred.yaml")
def predict_pipeline_command(cfg: DictConfig) -> None:
    cfg_yaml = OmegaConf.to_yaml(cfg)
    save_config(cfg_yaml)
    logger.info(f"Config params:")
    logger.info(cfg_yaml)

    params = read_prediction_pipeline_params(cfg)
    predict_pipeline(params)


if __name__ == "__main__":
    predict_pipeline_command()
