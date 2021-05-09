import logging
import sys

import numpy as np
import requests

from data import generate_dataset

SIZE = 100
NAN_PROBA = 0.05

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def main():
    data = generate_dataset(SIZE, NAN_PROBA)
    request_features = list(data.columns)
    for i in range(SIZE):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
        ]
        logger.debug(f"Input data: {request_data}")
        response = requests.get(
            "http://0.0.0.0:8000/predict/",
            json={"data": [request_data], "features": request_features},
        )
        logger.debug(f"Response code {response.status_code}. Result: {response.json()}")


if __name__ == "__main__":
    main()
