from sklearn.base import TransformerMixin, BaseEstimator


class RankTransformer(TransformerMixin, BaseEstimator):
    """
    Transforms columns to rank representation
    """
    def __init__(self, method='average', ascending=True):
        self.method = method
        self.ascending = ascending
        self.__mapping = {}

    def fit(self, X, y=None):
        for col in X.columns:
            rank = X[col].rank(method=self.method, ascending=self.ascending)
            self.__mapping[col] = dict(zip(X[col], rank))
        return self

    def transform(self, X):
        for col in X.columns:
            X[col] = X[col].map(self.__mapping[col])
        return X

    def get_mapping(self):
        return self.__mapping
