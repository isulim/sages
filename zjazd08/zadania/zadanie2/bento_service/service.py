import bentoml
import pandas as pd
from pydantic import BaseModel
from bentoml.io import JSON


model_runner = bentoml.picklable_model.get("netflixprophetmodel:latest").to_runner()

srv = bentoml.Service("NetflixProphetModel", runners=[model_runner])


class ModelFeatures(BaseModel):
    date: str


@srv.api(input=JSON(pydantic_model=ModelFeatures), output=JSON())
def predict(input_value: ModelFeatures) -> dict:
    input_series = pd.DataFrame([input_value.date], columns=['ds'])

    result = model_runner.run(input_series)

    print(result)

    result['ds'] = result['ds'].astype(str)
    return result
