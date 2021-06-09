from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from entities.scorer_params import ScorerParams

Scorer = Union[roc_auc_score, accuracy_score, f1_score]
Classifier = Union[LogisticRegression, RandomForestClassifier]


def calc_metric(X: pd.DataFrame, y: pd.Series, clf: Classifier, scorer: Scorer, threshold=None) -> float:
    y_pred = clf.predict_proba(X)[:, 1]
    if threshold is not None:
        y_pred = (y_pred > threshold).astype(int)
    score = scorer(y, y_pred)
    return score


def score_model(X: pd.DataFrame, y: pd.Series, clf: Classifier, cfg: ScorerParams) -> dict:
    scorer = globals()[cfg.scorer]
    score = calc_metric(X, y , clf, scorer, cfg.threshold)
    return {cfg.scorer: score}
