import bentoml
import dvc.api
from prophet.serialize import model_from_json

with dvc.api.open("models/prophet/model.json", remote="storage", rev="fbc15d3") as f:
    model_json = f.read()

model = model_from_json(model_json)

bentoml.picklable_model.save_model("NetflixProphetModel", model,
                                   signatures={"predict": {"batchable": False}})
