from GitMarco.tf import utils, metrics, basic
import numpy as np

from GitMarco.tf.losses import chamfer_distance
from GitMarco.tf.pointnet import Pointnet
from GitMarco.tf.utils import limit_memory, random_dataset


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


def test_chamfer_loss():
    x = utils.random_dataset(shape=(32, 1024, 3))
    y = utils.random_dataset(shape=(32, 1024, 3))
    chamfer_distance(x, y)


def test_pointnet():
    data = random_dataset(shape=(32, 4096, 3))
    test_data = random_dataset(shape=(32, 4096, 3))
    field = random_dataset(shape=(32, 4096, 2))
    field_test = random_dataset(shape=(32, 4096, 2))

    model = Pointnet(n_points=4096,)
    model = model.create_model()
    # model.model_2_image()
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    model.evaluate(data, field)
