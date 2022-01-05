from GitMarco.tf import utils, metrics, basic
import numpy as np

from GitMarco.tf.utils import limit_memory


def test_limit_memory():
    limit_memory()


def test_random_dataset():
    utils.random_dataset()


def test_r_squared():
    y = np.random.rand(100)
    predictions = np.random.rand(100)
    metrics.r_squared(y, predictions)


def test_basic_dense_model():
    model = basic.basic_dense_model(input_shape=(10,),
                                    output_shape=1,
                                    optimizer='adadelta')
    model.summary()





