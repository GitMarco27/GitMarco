import numpy as np
from GitMarco.graphics.plotly import Scatter3D


def test_scatter_3d():
    x = y = np.arange(1000)
    z = np.random.rand(1000)
    plot = Scatter3D(x=x,
                     y=y,
                     z=z,)
    fig = plot.plot(show=False)

