from typing import List

import numpy as np
import pandas as pd

from .predictor import Predictor
from .structs import OutputStruct


def make_sample(data: List, features: List[str]) -> pd.DataFrame:
    data = pd.DataFrame(np.array(data).reshape(1, -1), columns=features)
    # data = pd.DataFrame(data, columns=features)
    return data


def make_response(predictions: np.ndarray) -> List[OutputStruct]:
    response = [OutputStruct(predicted_class=pred) for pred in predictions]
    return response


def make_prediction(
    data: List, features: List, predictor: Predictor
) -> List[OutputStruct]:
    data = make_sample(data, features)
    predictions = predictor.predict_pipeline(data)
    response = make_response(predictions)
    return response
