import pandas as pd


def load_data(df_path):
    return pd.read_csv(df_path)


def x_extract(df):
    X_data = df[["passenger_id", "pclass", "sex", "age", "fare", "embarked"]]
    return X_data


def y_extract(df):
    y_data = df["survived"]
    return y_data


def x_y_extract(df):
    X = x_extract(df)
    y = y_extract(df)
    return X, y
