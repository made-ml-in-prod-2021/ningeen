from dataclasses import dataclass, field
from .split_params import SplittingParams
from .feature_params import FeatureParams
from .train_params import TrainingParams
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    input_data_url: str
    output_model_path: str
    metric_path: str
    transformer_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    version: str = field(default="0.0")


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(cfg: DictConfig) -> TrainingPipelineParams:
    schema = TrainingPipelineParamsSchema()
    params = schema.load(cfg)
    return params
