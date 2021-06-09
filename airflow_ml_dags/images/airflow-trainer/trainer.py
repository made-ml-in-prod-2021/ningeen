from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from entities.trainer_params import TrainerParams

Classifier = Union[LogisticRegression, RandomForestClassifier]


def train_model(X: pd.DataFrame, y: pd.Series, cfg: TrainerParams) -> Classifier:
    clf = globals()[cfg.clf](**cfg.clf_params)
    clf.fit(X.values, y.values.ravel())
    return clf
