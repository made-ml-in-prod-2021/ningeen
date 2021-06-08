from dataclasses import dataclass

import yaml
from marshmallow_dataclass import class_schema


@dataclass()
class PredictionPipelineParams:
    transformer_path: str
    model_path: str


PredictionPipelineParamsSchema = class_schema(PredictionPipelineParams)


def read_prediction_pipeline_params(path: str) -> PredictionPipelineParams:
    with open(path, "r") as input_stream:
        schema = PredictionPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
