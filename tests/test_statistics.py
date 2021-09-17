import numpy as np
from GitMarco.statistics.metrics import standard_error


def test_standard_error():
    data = np.random.rand(20)
    error = standard_error(data)
    assert isinstance(error, float), 'Error in standard error type'
