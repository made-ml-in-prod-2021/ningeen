import sys
import json

import click
import logging
import logging.config

from data import read_data, split_train_val_data
from features import build_transformer, extract_target, make_features, save_transformer
from models import (
    get_model,
    train_model,
    serialize_model,
    predict_model,
    evaluate_model,
)
from entities import TrainingPipelineParams, read_training_pipeline_params

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"Start training with params: {training_pipeline_params}")

    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"Raw data shape: {data.shape}")

    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"Train df shape: {train_df.shape}")
    logger.info(f"Val df shape: {val_df.shape}")

    pipeline = build_transformer(training_pipeline_params.feature_params)
    pipeline.fit(train_df)
    logger.info(f"Transform fitted.")

    train_features = make_features(pipeline, train_df)
    train_target = extract_target(
        train_df, training_pipeline_params.feature_params
    )
    logger.info(f"Train features shape: {train_features.shape}")

    val_features = make_features(pipeline, val_df)
    val_target = extract_target(
        val_df, training_pipeline_params.feature_params
    )
    logger.info(f"Val features shape: {train_features.shape}")

    model = get_model(training_pipeline_params.train_params)
    model = train_model(train_features, train_target, model)
    logger.info(f"Model trained.")

    predictions = predict_model(val_features, model)

    metrics = evaluate_model(predictions, val_target)

    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.info(f"Metrics are {metrics}")

    path_to_model = serialize_model(model, training_pipeline_params.output_model_path)
    logger.info(f"Model saved in {path_to_model}")

    path_to_transformer = save_transformer(pipeline, training_pipeline_params.transformer_path)
    logger.info(f"Transformer saved in {path_to_transformer}")

    return path_to_model, metrics


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
