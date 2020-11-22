import click
import pandas as pd


def chose_columns(df, columns):
    df_with_columns = df[columns]
    return df_with_columns


@click.group()
def cli():
    pass


@cli.command()
@click.argument("raw_data", type=click.Path())
@click.argument("prep_data", type=click.Path())
def clean_train(raw_data, prep_data):
    """
    "Очистка" тестовых данных
    """
    df = pd.read_csv(raw_data)
    columns = ["passenger_id", "pclass", "sex", "age", "fare", "embarked", "survived"]
    df_columns = chose_columns(df, columns)
    clean_df = df_columns.dropna()
    clean_df.to_csv(prep_data, index=False)


@cli.command()
@click.argument("raw_data", type=click.Path())
@click.argument("prep_data", type=click.Path())
def clean_test(raw_data, prep_data):
    """
    "Очистка" тестовых данных
    """
    df = pd.read_csv(raw_data)
    columns = ["passenger_id", "pclass", "sex", "age", "fare", "embarked"]
    df_columns = chose_columns(df, columns)
    clean_df = df_columns.dropna()
    clean_df.to_csv(prep_data, index=False)


if __name__ == "__main__":
    cli()
