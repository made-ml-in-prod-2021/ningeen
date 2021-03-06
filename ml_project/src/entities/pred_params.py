from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig


@dataclass()
class PredictionPipelineParams:
    data_path: str
    artifacts_dir: str
    transformer_path: str
    model_path: str
    output_path: str


PredictionPipelineParamsSchema = class_schema(PredictionPipelineParams)


def read_prediction_pipeline_params(cfg: DictConfig) -> PredictionPipelineParams:
    schema = PredictionPipelineParamsSchema()
    params = schema.load(cfg)
    return params
