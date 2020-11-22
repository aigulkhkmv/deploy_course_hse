from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from homework_1.scikitlearn_pipeline.preprocessors import ConnectFeatures

pipeline = Pipeline(
    steps=[
        ("connect", ConnectFeatures(all=True)),
        ("norm", Normalizer()),
        ("ada_boost_classifier", AdaBoostClassifier()),
    ]
)
