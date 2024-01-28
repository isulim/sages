import os

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

DATA = "data/cardiovascular.csv"

os.makedirs("model", exist_ok=True)
os.makedirs("data/ready", exist_ok=True)

df = pd.read_csv(DATA)
X = df.drop(["patientid", "target"], axis=1)
y = df["target"]

encoder_chest = OneHotEncoder()
encoder_electro = OneHotEncoder()
encoder_slope = OneHotEncoder()

x_chest = encoder_chest.fit_transform(X[["chestpain"]]).toarray()
x_electro = encoder_electro.fit_transform(X[["restingrelectro"]]).toarray()
x_slope = encoder_slope.fit_transform(X[["slope"]]).toarray()

X = X.drop(["chestpain", "restingrelectro", "slope"], axis=1)
X = pd.concat([
    X,
    pd.DataFrame(x_chest, columns=encoder_chest.get_feature_names_out()),
    pd.DataFrame(x_electro, columns=encoder_electro.get_feature_names_out()),
    pd.DataFrame(x_slope, columns=encoder_slope.get_feature_names_out()),

], axis=1)

X.to_csv("data/ready/X_ready.csv", index=False)
y.to_csv("data/ready/y_ready.csv", index=False)
