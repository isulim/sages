import os
import pytest
import numpy as np
import pandas as pd


def test_file_exists():
    assert os.path.isfile("../data/ready/X_ready.csv")
    assert os.path.isfile("../data/ready/y_ready.csv")


@pytest.fixture
def df_X():
    return pd.read_csv("../data/ready/X_ready.csv")


@pytest.fixture
def df_y():
    return pd.read_csv("../data/ready/y_ready.csv")


def test_X_shape(df_X):
    assert df_X.shape == (1000, 20)


def test_y_shape(df_y):
    assert df_y.shape == (1000, 1)


def test_X_columns(df_X):
    assert df_X.columns.to_list() == ['age',
                                   'gender',
                                   'restingBP',
                                   'serumcholestrol',
                                   'fastingbloodsugar',
                                   'maxheartrate',
                                   'exerciseangia',
                                   'oldpeak',
                                   'noofmajorvessels',
                                   'chestpain_0',
                                   'chestpain_1',
                                   'chestpain_2',
                                   'chestpain_3',
                                   'restingrelectro_0',
                                   'restingrelectro_1',
                                   'restingrelectro_2',
                                   'slope_0',
                                   'slope_1',
                                   'slope_2',
                                   'slope_3']


def test_dtypes(df_X):
    assert df_X.dtypes.isin([np.dtype('int64'), np.dtype('float64')]).all()
