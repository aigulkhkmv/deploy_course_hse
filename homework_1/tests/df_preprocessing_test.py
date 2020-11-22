import pandas as pd
from hypothesis import given
from hypothesis.extra.pandas import column, data_frames

from homework_1.procedure_pipeline.load_and_save_dataset import x_extract, y_extract, x_y_extract


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
