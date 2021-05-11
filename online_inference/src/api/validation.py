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
        self.values = dict(zip(self.rfeatures, self.rdata))

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
        if self.values["age"] <= 0 or self.values["age"] >= 100:
            raise HTTPException(
                status_code=INCORRECT_VALIDATION_STATUS_CODE,
                detail=f"Incorrect age: {self.values['age']}",
            )
        if self.values["sex"] < 0 or self.values["sex"] > 1:
            raise HTTPException(
                status_code=INCORRECT_VALIDATION_STATUS_CODE,
                detail=f"Incorrect sex: {self.values['sex']}",
            )
