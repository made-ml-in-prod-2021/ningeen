import logging
import os
import sys
from typing import List

import uvicorn
from fastapi import FastAPI

from api import Predictor, InputStruct, OutputStruct, make_prediction

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
predictor: Predictor = None

DEFAULT_PATH = "configs/config_pred.yaml"

app = FastAPI()


@app.get("/")
def main():
    return "Hello there"


@app.on_event("startup")
def load_predictor():
    global predictor
    params_path = os.getenv("PATH_TO_PARAMS")
    if params_path is None:
        logger.warning("No PATH_TO_PARAMS found, using default path.")
        params_path = DEFAULT_PATH

    predictor = Predictor.create_predictor(params_path)


@app.get("/predict/", response_model=List[OutputStruct])
def predict(request: InputStruct):
    response = make_prediction(request.data, request.features, predictor)
    return response


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
