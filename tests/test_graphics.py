import numpy as np
from GitMarco.graphics.plotly import Scatter3D, mesh_3d
from GitMarco.graphics.matplotlib import validation_plot, plot_2d, scatter_2d, circle


def test_scatter_3d():
    x = y = [np.arange(1000)]*2
    z = [np.random.rand(1000), np.random.rand(1000)-1]
    plot = Scatter3D(x=x,
                     y=y,
                     z=z, )
    fig = plot.plot(show=False, color=z, cmax=1, x_range=[0, 1000], y_range=[0, 1000], cmin=-1)
    # fig.show()


def test_mesh_3d():
    x = np.asarray([0, 1, 2, 0])
    y = np.asarray([0, 0, 1, 2])
    z = np.asarray([0, 2, 0, 1])
    i = np.asarray([0, 0, 0, 1])
    j = np.asarray([1, 2, 3, 2])
    k = np.asarray([2, 3, 1, 3])

    fig = mesh_3d(x, y, z, i, j, k,
                  show=False, color=[0, 0.33, 0.66, 1], title='Mesh3D', size=(800, 600),
                  showscale=True, show_axis=True, cmin=0, cmax=1,
                  colorscale=[[0, 'gold'], [0.5, 'mediumturquoise'], [1, 'magenta']],
                  x_range=[0, 2], y_range=[0, 2], z_range=[0, 2])


def test_validation_plot():
    true = np.random.rand(100)
    pred = np.random.rand(100)
    fig = validation_plot(true, pred, show=False)


def test_plot_2d():
    x = y = np.random.rand(1000)
    plot_2d(x, y, show=False, xlim=(-0.5, 1.5),
            line_width=3)
    x = [np.random.rand(1000), np.random.rand(1000)]
    y = [x[0] - 1, x[1] + 1]
    plot_2d(x, y, show=False,
            line_width=3, label=['1', '2'])


def test_scatter_2d():
    x = y = np.random.rand(1000)
    scatter_2d(x, y, show=False, xlim=(-0.5, 1.5))
    x = [np.random.rand(1000), np.random.rand(1000)]
    y = [x[0] - 1, x[1] + 1]
    scatter_2d(x, y, show=False, label=['1', '2'])


def test_circle():
    crc = circle(show=False, r=2, c=(1, 0), n=20)
