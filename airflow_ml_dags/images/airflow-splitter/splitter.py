import pandas as pd
from sklearn.model_selection import train_test_split

from entities.splitter_params import SplitterParams


def split_data(
    df: pd.DataFrame, target: pd.Series, cfg: SplitterParams
) -> (pd.DataFrame, pd.DataFrame, pd.Series, pd.Series):
    stratify = target if cfg.to_stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        df,
        target,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        shuffle=cfg.shuffle,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test
