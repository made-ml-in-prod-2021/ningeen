import sys
import json
import pickle

import click
import logging
import logging.config
import numpy as np
import pandas as pd

from data import read_data, split_train_val_data
from features import build_transformer, extract_target, make_features, load_transformer
from models import (
    get_model,
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
    load_model
)
from entities import TrainingPipelineParams, read_training_pipeline_params

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def save_prediction(predictions: np.ndarray, output_path: str) -> str:
    pd.Series(predictions).to_csv(output_path, index=False)
    return output_path


def predict_pipeline(data_path: str, transformer_path: str, model_path: str, output_path: str):
    logger.info(
        f"Start prediction with paths: data_path: {data_path}, "
        f"transformer_path: {transformer_path}, model_path: {model_path}"
    )

    data = read_data(data_path)
    logger.info(f"Data loaded. Raw data shape: {data.shape}")

    pipeline = load_transformer(transformer_path)
    logger.info(f"Transformer loaded: {pipeline}")

    model = load_model(model_path)
    logger.info(f"Model loaded: {model}")

    train_features = make_features(pipeline, data)
    logger.info(f"Test features shape: {train_features.shape}")

    predictions = predict_model(train_features, model)
    predictions_path = save_prediction(predictions, output_path)
    logger.info(f"Predictions saved in {predictions_path}")


@click.command(name="predict_pipeline")
@click.argument("data_path")
@click.argument("transformer_path")
@click.argument("model_path")
@click.argument("output_path")
def predict_pipeline_command(data_path: str, transformer_path: str, model_path: str, output_path: str):
    predict_pipeline(data_path, transformer_path, model_path, output_path)


if __name__ == "__main__":
    predict_pipeline_command()
