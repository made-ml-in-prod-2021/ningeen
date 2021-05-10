import sys
from fastapi.testclient import TestClient
from src.app import app
from src.data import generate_dataset
from src import features

# https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
sys.modules['features'] = features

SIZE = 5
NAN_PROBA = 0.05


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

            response = client.get(
                "http://0.0.0.0:8000/predict/",
                json={"data": [request_data], "features": request_features},
            )

            assert response.status_code == 200
            result = response.json()
            assert len(result) == 1
            result = result[0]
            assert "predicted_class" in result
            assert result["predicted_class"] in [0, 1]
