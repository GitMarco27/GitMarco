from GitMarco.tf import utils, metrics, basic
import numpy as np
from GitMarco.tf.losses import chamfer_distance, euclidian_dist_loss
from GitMarco.tf.optimization import OptiLoss, GradientOptimizer
from GitMarco.tf.pointnet import Pointnet
from GitMarco.tf.utils import limit_memory, random_dataset
import pandas as pd
from GitMarco.tf.basic import basic_dense_model
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


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


def test_euclidean_distance():
    x = utils.random_dataset(shape=(32, 1024, 3))
    y = utils.random_dataset(shape=(32, 1024, 3))
    euclidian_dist_loss(x, y, correction=True)


class Loss(OptiLoss):
    def __init__(self, params):
        super(Loss, self).__init__(params)

    def __call__(self, sample):
        return self.model(sample)[0][0]


def test_gradient_optimizer():
    with tf.device('CPU:0'):
        df = pd.DataFrame(random_dataset(shape=(32, 4)).numpy())
        df.columns = ['x1', 'x2', 'y1', 'y2']
        model = basic_dense_model(input_shape=(2,), output_shape=2)
        model.compile(optimizer='Adam')
        model.fit(random_dataset(shape=(32, 2)), random_dataset(shape=(32, 2)), epochs=1)

        optimizer = GradientOptimizer(
            model,
            df,
            StandardScaler(),
            Loss(dict(model=model)),
            n_features=2,
            n_labels=2,
            iterations=10

        )

        optimizer.run()
        optimizer.history()
        optimizer.get_best_sample()
        optimizer.get_results()
        optimizer.compare_bounds()
        optimizer.reset()
        optimizer.iterations = 100
        optimizer.run()
        optimizer.history()
        optimizer.get_best_sample()
        optimizer.get_results()




