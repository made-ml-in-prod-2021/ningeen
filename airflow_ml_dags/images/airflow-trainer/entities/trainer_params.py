from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from omegaconf import DictConfig


@dataclass()
class TrainerParams:
    model_file_name: str
    clf: str
    clf_params: dict


def read_trainer_params(cfg: DictConfig) -> TrainerParams:
    trainer_schema = class_schema(TrainerParams)
    schema = trainer_schema()
    params = schema.load(cfg)
    return params
