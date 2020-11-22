import tempfile

import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

from homework_1.procedure_pipeline.train import train_model


# подумать, как лучше сделать с временными путями, хочу сразу передавать временный путь...
@given(tuple(arrays(np.float, 3, elements=floats(0, 1)), arrays(np.float, 1, elements=int(0, 1))))
def test_train_model(X_train, y_train, model_path, normalizer_path):
    tmp_dir = tempfile.TemporaryDirectory()
    train_model()
    X_train, y_train, model_path, normalizer_path
