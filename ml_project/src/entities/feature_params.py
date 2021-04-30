from dataclasses import dataclass, field
from typing import List, Optional


@dataclass()
class FeatureParams:
    target_col: Optional[str]
    categorical_features: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    rank_features: List[str] = field(default_factory=list)
    features_to_drop: List[str] = field(default_factory=list)
    method: Optional[str] = field(default="average")
    ascending: Optional[bool] = field(default=True)
    normalize: Optional[bool] = field(default=False)
