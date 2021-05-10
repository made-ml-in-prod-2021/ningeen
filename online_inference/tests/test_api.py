import sys
import random
from fastapi.testclient import TestClient
from src.app import app
from src.data import generate_dataset
from src import features

# https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
sys.modules['features'] = features

SIZE = 5
NAN_PROBA = 0.05
PREDICT_URL = "http://0.0.0.0:8000/predict/"


def get_response(client, request_data, request_features):
    response = client.get(
        PREDICT_URL,
        json={"data": [request_data], "features": request_features},
    )
    return response


def test_read_main():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == "Hello there"


def test_predict():
    with TestClient(app) as client:
        data = generate_dataset(SIZE, NAN_PROBA)
        request_features = list(data.columns)
        for i in range(SIZE):
            request_data = data.iloc[i].tolist()
            response = get_response(client, request_data, request_features)
            assert response.status_code == 200
            result = response.json()
            assert len(result) == 1
            result = result[0]
            assert "predicted_class" in result
            assert result["predicted_class"] in [0, 1]


def test_validation_length():
    with TestClient(app) as client:
        data = generate_dataset(SIZE, NAN_PROBA)

        request_data = data.iloc[0].tolist()
        request_features = list(data.columns)
        response = get_response(client, request_data[:-1], request_features)
        assert response.status_code == 422

        response = get_response(client, request_data, request_features[:-1])
        assert response.status_code == 422

        response = get_response(client, request_data[:-1], request_features[:-1])
        assert response.status_code == 422


def test_validation_columns():
    with TestClient(app) as client:
        data = generate_dataset(SIZE, NAN_PROBA)

        request_data = data.iloc[0].tolist()
        request_features = list(data.columns)

        for i in range(SIZE):
            random.shuffle(request_features)
            if request_features != list(data.columns):
                response = get_response(client, request_data, request_features)
                assert response.status_code == 400


def test_validation_values():
    with TestClient(app) as client:
        data = generate_dataset(SIZE, NAN_PROBA)
        request_features = list(data.columns)

        for i in range(2):
            request_data = data.iloc[0].tolist()
            request_data[i] = 120
            response = get_response(client, request_data, request_features)
            assert response.status_code == 400
