import mlflow
import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://127.0.0.1:5000")


@pytest.fixture
def loaded_model():
    model = mlflow.sklearn.load_model("models:/Random Forest/1")
    return model


def test_load_model(loaded_model):
    assert isinstance(loaded_model, RandomForestClassifier)


@pytest.fixture
def pd_x():
    X = pd.DataFrame([{
        "age": 59,
        "gender": 1,
        "restingBP": 182,
        "serumcholestrol": 177,
        "fastingbloodsugar": 0,
        "maxheartrate": 168,
        "exerciseangia": 0,
        "oldpeak": 2.1,
        "noofmajorvessels": 1,
        "chestpain_0": 0.0,
        "chestpain_1": 0.0,
        "chestpain_2": 1.0,
        "chestpain_3": 0.0,
        "restingrelectro_0": 0.0,
        "restingrelectro_1": 1.0,
        "restingrelectro_2": 0.0,
        "slope_0": 0.0,
        "slope_1": 0.0,
        "slope_2": 1.0,
        "slope_3": 0.0
    }])
    return X


def test_predict(loaded_model, pd_x):
    y = loaded_model.predict(pd_x)
    assert y == 1
