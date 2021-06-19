from dataclasses import dataclass, field

from marshmallow_dataclass import class_schema
from omegaconf import DictConfig


@dataclass()
class ScorerParams:
    scorer: str
    threshold: float = field(default=None)


def read_scorer_params(cfg: DictConfig) -> ScorerParams:
    scorer_schema = class_schema(ScorerParams)
    schema = scorer_schema()
    params = schema.load(cfg)
    return params
