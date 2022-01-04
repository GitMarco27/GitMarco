import numpy as np
from GitMarco.graphics.plotly import Scatter3D
from GitMarco.graphics.matplotlib import validation_plot, plot_2d, scatter_2d


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


def test_plot_2d():
    x = y = np.random.rand(1000)
    plot_2d(x, y, show=False, xlim=(-0.5, 1.5),
            line_width=3)
    x = [np.random.rand(1000), np.random.rand(1000)]
    y = [x[0]-1, x[1]+1]
    plot_2d(x, y, show=False,
            line_width=3, label=['1', '2'])


def test_scatter_2d():
    x = y = np.random.rand(1000)
    scatter_2d(x, y, show=False, xlim=(-0.5, 1.5))
    x = [np.random.rand(1000), np.random.rand(1000)]
    y = [x[0]-1, x[1]+1]
    scatter_2d(x, y, show=False, label=['1', '2'])