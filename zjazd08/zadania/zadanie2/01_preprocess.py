import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = "data/NFLX.csv"


def preprocess_data(train_output_file, val_output_file):
    os.makedirs(os.path.dirname(train_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(val_output_file), exist_ok=True)

    df = pd.read_csv(DATA)
    df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)

    pd.DataFrame(df_train).to_csv(train_output_file, index=False)
    pd.DataFrame(df_test).to_csv(val_output_file, index=False)


if __name__ == "__main__":
    train_output_file = sys.argv[1]
    val_output_file = sys.argv[2]

    preprocess_data(train_output_file, val_output_file)
