import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.extra.pandas import column, data_frames
from hypothesis.strategies import floats, integers, sampled_from

from homework_1.procedure_pipeline.load_and_save_dataset import x_extract, y_extract, x_y_extract
from homework_1.procedure_pipeline.preprocessing_functions import text2num_sex
from homework_1.procedure_pipeline.train import train_model


@given(
    data_frames(
        [
            column("passenger_id", dtype=int),
            column("pclass", dtype=int),
            column("sex", dtype=str),
            column("age", dtype=int),
            column("fare", dtype=float),
            column("embarked", dtype=str),
            column("some label", dtype=int),
        ]
    )
)
def test_check_x_extract(df):
    assert x_extract(df).shape[1] == 6


@given(
    data_frames(
        [
            column("passenger_id", dtype=int),
            column("pclass", dtype=int),
            column("sex", dtype=str),
            column("age", dtype=int),
            column("fare", dtype=float),
            column("embarked", dtype=str),
            column("survived", dtype=int),
        ]
    )
)
def test_check_y_extract(df):
    assert type(y_extract(df)) == pd.core.series.Series


@given(
    data_frames(
        [
            column("passenger_id", dtype=int),
            column("pclass", dtype=int),
            column("sex", dtype=str),
            column("age", dtype=int),
            column("fare", dtype=float),
            column("embarked", dtype=str),
            column("survived", dtype=int),
        ]
    )
)
def test_check_x_y_extract(df):
    assert len(x_y_extract(df)) == 2


@given(
    arrays(np.float, (5, 6), elements=floats(0, 1)),
    arrays(np.int8, (5, 1), elements=integers(0, 1)),
)
def test_train_model(X_train, y_train):
    assert len(train_model(X_train, y_train.reshape(-1, 1))) == 2


@given(
    data_frames(
        [
            column("passenger_id", dtype=int),
            column("pclass", dtype=int),
            column("sex", dtype=str, elements=sampled_from(["male", "female"])),
            column("age", dtype=int),
            column("fare", dtype=float),
            column("embarked", dtype=str),
            column("survived", dtype=int),
        ]
    )
)
def test_text2num_sex(df):
    answ = text2num_sex(df)
    assert len(set(answ["sex"])) == 0 or len(set(answ["sex"])) == 1 or len(set(answ["sex"])) == 2
