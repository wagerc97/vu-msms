import numpy as np
from scipy import constants

class AnisotropyField(object):
    def __init__(self, mesh, material):
        self._mesh = mesh
        self._Ms = material["Ms"]
        self._K = material["K"]
        self._K_axis = np.array(material["K_axis"]).reshape(1,1,1,-3)

    def h(self, t, m):
        # TODO
        # Implement anisotropy field:
        # 2 * K / (mu_0 * Ms) * K_axis * <K_axis, m>
        raise NotImplementedError

    def E(self, t, m):
        # TODO
        # Implement anisotropy energy:
        raise NotImplementedError
