from src.data.make_dataset import read_data, split_train_val_data
from src.entities import SplittingParams


def test_load_dataset(fake_dataset, dataset_size):
    data = read_data(fake_dataset)
    assert len(data) == dataset_size


def test_split_dataset(fake_dataset, dataset_size, config_test_fixture):
    splitting_params = SplittingParams(
        random_state=config_test_fixture.splitting_random_state,
        val_size=config_test_fixture.splitting_val_size,
    )
    data = read_data(fake_dataset)
    train, val = split_train_val_data(data, splitting_params)
    assert len(train) + len(val) == dataset_size
    assert len(train) > 0
    assert len(val) > 0
