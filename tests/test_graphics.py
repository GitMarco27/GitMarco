import numpy as np
from GitMarco.graphics.plotly import Scatter3D
from GitMarco.graphics.matplotlib import validation_plot


def test_scatter_3d():
    x = y = np.arange(1000)
    z = np.random.rand(1000)
    plot = Scatter3D(x=x,
                     y=y,
                     z=z,)
    fig = plot.plot(show=False)


def test_validation_plot():
    true = np.random.rand(100)
    pred = np.random.rand(100)
    fig = validation_plot(true, pred, show=False)

