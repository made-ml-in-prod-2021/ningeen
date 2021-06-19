from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from typing import List, Dict
from omegaconf import DictConfig


@dataclass()
class DataParams:
    column_names: List[str]
    dataset_size: int
    probability: float
    target_col: str
    target_threshold: int
    threshold_col: str
    limits: Dict[str, List[int]]


def read_data_params(cfg: DictConfig) -> DataParams:
    data_schema = class_schema(DataParams)
    schema = data_schema()
    params = schema.load(cfg)
    return params
