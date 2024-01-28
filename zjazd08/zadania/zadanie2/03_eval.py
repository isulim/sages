from sklearn.metrics import mean_squared_error
import sys
import pandas as pd
from dvclive import Live
from prophet.serialize import model_from_json


def eval(val_input_file):
    with Live("eval") as live:
        model_path = "models/prophet/model.json"

        with open(model_path, "r") as f:
            model_json = f.read()

        model = model_from_json(model_json)

        val_data = pd.read_csv(val_input_file)

        future = val_data[['Date']].copy().rename(columns={'Date': 'ds'})
        forecast = model.predict(future)

        actual_data = val_data['Close'].values
        predicted_data = forecast['yhat'].values

        mse = mean_squared_error(actual_data, predicted_data)
        print(f'MSE: {mse}')

        live.log_metric("mse", mse)
        live.log_artifact(model_path, type="model", name="prophet")


if __name__ == "__main__":
    val_input_file = sys.argv[1]
    eval(val_input_file)
