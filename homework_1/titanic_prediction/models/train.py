import json
import pickle

import click
import pandas as pd
from loguru import logger
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from homework_1.titanic_prediction.features.feature_creation import ConnectFeatures


@click.command()
@click.argument("train_path", type=click.File("rb", lazy=True))
@click.argument("test_path", type=click.File("rb", lazy=True))
@click.argument("test_y_path", type=click.Path())
@click.argument("metrics_path", type=click.File("w", lazy=True))
@click.argument("model_path", type=click.File("wb", lazy=True))
def train(train_path, test_path, test_y_path, metrics_path, model_path):
    df = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    y_test_ = pd.read_csv(test_y_path)

    y_train = df["survived"]
    X_train = df[["passenger_id", "pclass", "sex", "age", "fare", "embarked"]]
    y_test = y_test_[y_test_["passenger_id"].isin(X_test["passenger_id"].to_list())]["survived"]

    pipeline = Pipeline(
        steps=[
            ("connect", ConnectFeatures(all=True)),
            ("norm", Normalizer()),
            ("ada_boost_classifier", AdaBoostClassifier()),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_train_predict = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    metrics = {
        "train_recall": recall_score(y_train, y_train_predict),
        "train_precision": precision_score(y_train, y_train_predict),
        "test_recall": recall_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred),
    }

    logger.info(
        "Train metrics: precision {}, recall {}",
        metrics["train_precision"],
        metrics["train_recall"],
    )
    logger.info(
        "Test metrics: precision {}, recall {}", metrics["test_precision"], metrics["test_recall"]
    )

    json.dump(metrics, metrics_path)
    pickle.dump(pipeline, model_path)


if __name__ == "__main__":
    train()
