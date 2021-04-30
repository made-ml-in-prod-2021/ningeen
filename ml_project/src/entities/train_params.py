from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="RandomForestClassifier")
    rf_params: dict = field(default={"random_state": 42})
    lr_params: dict = field(default={"random_state": 42})
