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


# from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
# Scorer = Union[roc_auc_score, accuracy_score, f1_score]
# def calc_metric(X: pd.DataFrame, y: pd.Series,  clf: Classifier, scorer: Scorer, threshold=None) -> float:
#     y_pred = clf.predict_proba(X)
#     if threshold is not None:
#         y_pred = (y_pred > threshold).astype(int)
#     score = scorer(y, y_pred)
#     return score
