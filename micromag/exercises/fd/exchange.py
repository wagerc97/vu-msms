import numpy as np
from scipy import ndimage, constants

class ExchangeField(object):
    def __init__(self, mesh, material):
        self._mesh = mesh
        self._A = material["A"]
        self._Ms = material["Ms"]

        # initialize laplace kernel
        self._kernel = np.zeros((3,3,3))
        # TODO setup 3D Laplace kernel

        # initialize scratch space
        self._h = np.zeros(mesh.n + (3,))

    def h(self, t, m):
        # TODO
        # Implement exchange field:
        # 2 * A / (mu_0 * Ms) * Laplace(m)
        # TIP: use ndimage.convolve with self._kernel
        raise NotImplementedError

    def E(self, t, m):
        # TODO
        # Implement exchange energy:
        raise NotImplementedError
