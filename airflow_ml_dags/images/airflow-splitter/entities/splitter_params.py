from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig
from typing import List


@dataclass()
class SplitterParams:
    shuffle: bool
    to_stratify: bool
    test_size: float
    random_state: int


def read_splitter_params(cfg: DictConfig) -> SplitterParams:
    splitter_schema = class_schema(SplitterParams)
    schema = splitter_schema()
    params = schema.load(cfg)
    return params
