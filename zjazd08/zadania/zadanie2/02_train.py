import sys
from pathlib import Path

import pandas as pd
import dvc.api

from prophet import Prophet, serialize
from sklearn.metrics import mean_squared_error


def train_model(train_data_file, val_data_file, params):
    train_data = pd.read_csv(train_data_file)
    test_data = pd.read_csv(val_data_file)

    train_data = train_data.rename(columns={'Date': 'ds', 'Close': 'y'})

    prophet_params = {
        "growth": params.get("train", {}).get("growth", "linear"),
        "seasonality_mode": params.get("train", {}).get("seasonality_mode", "additive"),
        "n_changepoints": params.get("train", {}).get("n_changepoints", 25),
    }

    model = Prophet(**prophet_params)
    model.fit(train_data)

    future = test_data[['Date']].copy().rename(columns={'Date': 'ds'})
    forecast = model.predict(future)

    actual_data = test_data['Close'].values
    predicted_data = forecast['yhat'].values

    mse = mean_squared_error(actual_data, predicted_data)
    print(f'MSE: {mse}')

    model_path = Path('models/prophet')
    model_path.mkdir(parents=True, exist_ok=True)

    with open(model_path / 'model.json', 'w') as f:
        json_model = serialize.model_to_json(model)
        f.write(json_model)


if __name__ == "__main__":

    params = dvc.api.params_show()
    print(params)

    train_data_file = sys.argv[1]
    val_data_file = sys.argv[2]

    train_model(train_data_file, val_data_file, params)
