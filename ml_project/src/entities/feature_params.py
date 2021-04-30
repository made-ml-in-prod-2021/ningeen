from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    rank_features: List[str]
    features_to_drop: List[str]
    target_col: Optional[str]
    method: Optional[str] = field(default="average")
    ascending: Optional[bool] = field(default=True)
