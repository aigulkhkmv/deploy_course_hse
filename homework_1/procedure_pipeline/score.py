import click

from homework_1.procedure_pipeline.load_and_save_dataset import load_data, x_y_extract
from homework_1.procedure_pipeline.preprocessing_functions import categorical2num, df2array
from homework_1.procedure_pipeline.train import train_model, train_metrics_count, save


@click.command()
@click.argument("train_path", type=click.File("rb", lazy=True))
@click.argument("test_x_path", type=click.File("rb", lazy=True))
@click.argument("test_y_path", type=click.Path())
@click.argument("metrics_path", type=click.File("w", lazy=True))
@click.argument("model_path", type=click.File("wb", lazy=True))
@click.argument("normalizer_path", type=click.File("wb", lazy=True))
def predict(train_path, test_x_path, test_y_path, metrics_path, model_path, normalizer_path):
    df = load_data(train_path)
    df = categorical2num(df)
    X_train, y_train = x_y_extract(df)
    X_train = df2array(X_train)

    model, normalizer = train_model(X_train, y_train.to_numpy())
    save(model, model_path)
    save(model, normalizer_path)
    train_metrics_count(
        model, normalizer, metrics_path, X_train, y_train.to_numpy(), test_x_path, test_y_path
    )


if __name__ == "__main__":
    predict()
