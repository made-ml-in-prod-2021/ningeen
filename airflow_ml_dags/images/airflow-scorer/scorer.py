from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from entities.scorer_params import ScorerParams

Scorer = Union[roc_auc_score, accuracy_score, f1_score]
Classifier = Union[LogisticRegression, RandomForestClassifier]


def calc_metric(y_true: pd.Series, y_pred: pd.Series, scorer: Scorer, threshold=None) -> float:
    if threshold is not None:
        y_pred = (y_pred > threshold).astype(int)
    score = scorer(y_true, y_pred)
    return score


def score_model(y_true: pd.Series, y_pred: pd.Series, cfg: ScorerParams) -> dict:
    scorer = globals()[cfg.scorer]
    score = calc_metric(y_true, y_pred, scorer, cfg.threshold)
    return {cfg.scorer: score}
