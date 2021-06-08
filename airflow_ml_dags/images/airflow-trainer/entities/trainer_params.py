from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig
from typing import List, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass()
class SplittingParams:
    splitter: str
    test_size: float
    random_state: int


@dataclass()
class PreprocessParams:
    scaler: str
    cols_to_drop: List[str]
    cols_to_scale: List[str]


@dataclass()
class PipelineParams:
    splitting_params: SplittingParams
    preprocess_params: PreprocessParams


def read_data_params(cfg: DictConfig) -> PipelineParams:
    preprocess_schema = class_schema(PipelineParams)
    schema = preprocess_schema()
    params = schema.load(cfg)
    return params
