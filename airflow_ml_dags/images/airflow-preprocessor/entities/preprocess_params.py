from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig
from typing import List


@dataclass()
class PreprocessParams:
    scaler: str
    cols_to_drop: List[str]
    cols_to_scale: List[str]


def read_preprocess_params(cfg: DictConfig) -> PreprocessParams:
    preprocess_schema = class_schema(PreprocessParams)
    schema = preprocess_schema()
    params = schema.load(cfg)
    return params
