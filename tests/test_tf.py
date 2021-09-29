from GitMarco.tf import utils, metrics
import numpy as np


def test_random_dataset():
    utils.random_dataset()


def test_r_squared():
    y = np.random.rand(100)
    predictions = np.random.rand(100)
    metrics.r_squared(y, predictions)
