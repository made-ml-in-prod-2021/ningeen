import numpy as np
import pytest

from src.data.make_dataset import read_data
from src.features import RankTransformer


@pytest.mark.parametrize(
    "method, ascending",
    [
        pytest.param('average', True, id='average_true'),
        pytest.param('average', False, id='average_false'),
        pytest.param('min', True, id='min'),
        pytest.param('first', True, id='first'),
    ]
)
def test_rank_init(method, ascending):
    transformer = RankTransformer(method, ascending)
    assert transformer.method == method
    assert transformer.ascending == ascending


@pytest.mark.parametrize(
    "method, ascending",
    [
        pytest.param('average', True, id='average_true'),
        pytest.param('average', False, id='average_false'),
        pytest.param('min', True, id='min'),
        pytest.param('first', True, id='first'),
    ]
)
def test_rank_fit_and_transform(fake_dataset, target_col, method, ascending):
    data = read_data(fake_dataset)
    X = data.drop(columns=[target_col])
    transformer = RankTransformer(method, ascending)
    assert len(transformer.get_mapping()) == 0
    res = transformer.fit(X)
    assert len(transformer.get_mapping()) > 0
    assert type(res) == RankTransformer
    X_rank = transformer.transform(X)
    assert X_rank.shape == X.shape
    assert not (X_rank.values == X.values).all().all()
