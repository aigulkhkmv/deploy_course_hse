import json
import pickle

from loguru import logger
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.preprocessing import Normalizer

from homework_1.procedure_pipeline.load_and_save_dataset import load_data
from homework_1.procedure_pipeline.preprocessing_functions import categorical2num, df2array


def train_model(X_train, y_train, model_path, normalizer_path):
    normalizer = Normalizer()
    normalizer.fit(X_train)
    normalized_data = normalizer.transform(X_train)

    ada_boost_classifier = AdaBoostClassifier()
    ada_boost_classifier.fit(normalized_data, y_train)

    pickle.dump(ada_boost_classifier, model_path)
    pickle.dump(normalizer, normalizer_path)

    return ada_boost_classifier, normalizer


def train_metrics_count(
    model, normalizer, metrics_path, X_train, y_train, test_x_path, test_y_path
):
    X_test_df = load_data(test_x_path)
    y_test_ = load_data(test_y_path)
    y_test = y_test_[y_test_["passenger_id"].isin(X_test_df["passenger_id"].to_list())]["survived"]
    X_test_df = categorical2num(X_test_df)
    X_test = df2array(X_test_df)

    y_train_predict = model.predict(normalizer.transform(X_train))
    y_test_predict = model.predict(normalizer.transform(X_test))

    metrics = {
        "train_recall": recall_score(y_train, y_train_predict),
        "train_precision": precision_score(y_train, y_train_predict),
        "test_recall": recall_score(y_test, y_test_predict),
        "test_precision": precision_score(y_test, y_test_predict),
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
