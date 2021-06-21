import logging
import os
import sys
import time
from typing import List

import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from src.api import Predictor, InputStruct, OutputStruct, make_prediction, Validator

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
predictor: Predictor = None

DEFAULT_PATH = "configs/config_pred.yaml"

app = FastAPI()
START_DELAY_SECONDS = 30
DEATH_DELAY_TIME = 60
START_TIME = time.time()


@app.get("/")
def main():
    return "Hello there"


@app.on_event("startup")
def load_predictor():
    global predictor
    time.sleep(START_DELAY_SECONDS)
    params_path = os.getenv("PATH_TO_PARAMS")
    if params_path is None:
        logger.warning("No PATH_TO_PARAMS found, using default path.")
        params_path = DEFAULT_PATH

    predictor = Predictor.create_predictor(params_path)


@app.get("/predict/", response_model=List[OutputStruct])
def predict(request: InputStruct):
    Validator(request).validate_input()
    response = make_prediction(request.data, request.features, predictor)
    return response


@app.get("/readiness")
def readiness() -> bool:
    return predictor is not None


@app.get("/liveness")
def liveness() -> PlainTextResponse:
    work_time = time.time() - START_TIME
    if work_time < DEATH_DELAY_TIME:
        response = PlainTextResponse("Live", status_code=200)
    else:
        response = PlainTextResponse("Dead", status_code=400)
    return response


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
