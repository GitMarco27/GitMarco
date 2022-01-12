import numpy as np
from scipy.interpolate import LinearNDInterpolator


def stl2mesh3d(stl_mesh):
    # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points)
    # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d

    p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)

    # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
    # extract unique vertices from all mesh triangles

    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0)
    i = np.take(ixr, [3 * k for k in range(p)])
    j = np.take(ixr, [3 * k + 1 for k in range(p)])
    k = np.take(ixr, [3 * k + 2 for k in range(p)])

    return vertices, i, j, k


def grid2grid_interp(grid, new_grid, feature, fill_value=None):

    if fill_value is None:
        interp = LinearNDInterpolator(list(zip(grid[:, 0], grid[:, 1], grid[:, 2])),
                                      feature,
                                      rescale=False,
                                      )
    else:
        interp = LinearNDInterpolator(list(zip(grid[:, 0], grid[:, 1], grid[:, 2])),
                                      feature,
                                      rescale=False,
                                      fill_value=fill_value
                                      )
    y = interp(new_grid[:, 0], new_grid[:, 1], new_grid[:, 2])

    return y
