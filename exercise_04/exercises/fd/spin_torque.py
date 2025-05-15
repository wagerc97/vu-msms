import numpy as np
from scipy import ndimage, constants

class SpinTorque(object):
    def __init__(self, mesh, material, j):
        self._stencil = np.zeros((3,3,3,3))
        # TODO
        # setup 3D kernel for grad

        self._mesh = mesh
        self._j = j
        self._gamma = material["gamma"]
        self._b = material["b"]
        self._xi = material["xi"]

        # initialize scratch spaces
        self._jgradm = np.zeros(mesh.n + (3,))
        #self._h = np.zeros(mesh.n + (3,))

    def h(self, t, m):
        # TODO
        # Implement Zhang-Li Spin-Torque Term
        # b/gamma [m x (j * grad(m)) + xi * j * grad(m)]
        # TIP: use ndimage.convolve with self._stencil to compute grad(m)
        raise NotImplementedError

    def E(self, t, m):
        # TODO
        # nothing, since Spin-Torque is a nonconservative term :)
        raise NotImplementedError
