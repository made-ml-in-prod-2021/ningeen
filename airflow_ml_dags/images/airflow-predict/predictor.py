from typing import Union

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

Classifier = Union[LogisticRegression, RandomForestClassifier]


def predict_proba(X: pd.DataFrame, clf: Classifier) -> pd.Series:
    y_pred = clf.predict_proba(X)[:, 1]
    y_pred = pd.Series(y_pred)
    return y_pred
