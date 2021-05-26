import pytest

from src.data.make_dataset import read_data
from src.features import RankTransformer


def test_rank_init(config_test_fixture):
    for method, ascending in config_test_fixture.rank_params.values():
        transformer = RankTransformer(method, ascending)
        assert transformer.method == method
        assert transformer.ascending == ascending


def test_rank_fit_and_transform(fake_dataset, target_col, config_test_fixture):
    data = read_data(fake_dataset)
    X = data.drop(columns=[target_col])
    for method, ascending in config_test_fixture.rank_params.values():
        transformer = RankTransformer(method, ascending)
        assert len(transformer.get_mapping()) == 0
        res = transformer.fit(X)
        assert len(transformer.get_mapping()) > 0
        assert type(res) == RankTransformer
        X_rank = transformer.transform(X)
        assert X_rank.shape == X.shape
        assert not (X_rank.values == X.values).all().all()
