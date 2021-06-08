from typing import List, Union

from pydantic import BaseModel, conlist

N_FEATURES = 14


class InputStruct(BaseModel):
    data: List[
        conlist(Union[int, float, None], min_items=N_FEATURES, max_items=N_FEATURES)
    ]
    features: conlist(str, min_items=N_FEATURES, max_items=N_FEATURES)


class OutputStruct(BaseModel):
    predicted_class: int
