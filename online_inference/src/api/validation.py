from fastapi import HTTPException

from .structs import InputStruct

INCORRECT_VALIDATION_STATUS_CODE = 400
N_FEATURES = 14
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


class Validator:
    def __init__(self, request: InputStruct) -> None:
        self.request = request
        self.rdata, self.rfeatures = request.data[0], request.features

    def validate_input(self) -> None:
        self.validate_columns()
        self.validate_values()

    def validate_columns(self) -> None:
        if self.rfeatures != COLUMNS:
            raise HTTPException(
                status_code=INCORRECT_VALIDATION_STATUS_CODE,
                detail=f"Incorrect feature names",
            )

    def validate_values(self) -> None:
        values = dict(zip(self.rfeatures, self.rdata))
        if values["age"] <= 0 or values["age"] >= 100:
            raise HTTPException(
                status_code=INCORRECT_VALIDATION_STATUS_CODE,
                detail=f"Incorrect age: {values['age']}",
            )
        if values["sex"] < 0 or values["sex"] > 1:
            raise HTTPException(
                status_code=INCORRECT_VALIDATION_STATUS_CODE,
                detail=f"Incorrect sex: {values['sex']}",
            )
