import pandas as pd

import mlflow
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")


model = mlflow.sklearn.load_model("models:/Random Forest/1")
print(model)

X = pd.read_csv("data/ready/X_ready.csv")
y = pd.read_csv("data/ready/y_ready.csv")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_val)
print(y_pred)
