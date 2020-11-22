from sklearn.base import BaseEstimator, TransformerMixin


class ConnectFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, all: bool = True):
        self.all = all

    def fit(self, X, y=None) -> "ConnectFeatures":
        return self

    def transform(self, X, y=None):
        if all:
            X["sex"] = X["sex"].replace("male", 0)
            X["sex"] = X["sex"].replace("female", 1)
            X["embarked"] = X["embarked"].replace("S", 0)
            X["embarked"] = X["embarked"].replace("C", 1)
            X["embarked"] = X["embarked"].replace("Q", 2)
            X = X.to_numpy()
            return X
        else:
            return Exception
