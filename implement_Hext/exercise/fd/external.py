import numpy as np
from scipy import constants

class ExternalField(object):
    def __init__(self, mesh, material, h):
        self._mesh = mesh
        self._Ms = material["Ms"]
        self._h = np.zeros(mesh.n + (3,))
        self._h[:,:,:,:] = np.array(h)      # h is a 4D tensor 

    def h(self, t, m):
        return self._h

    def E(self, t, m):
        # TODO: Implement Zeeman energy from the field (the field is given in the class)
        # - mu_0 \integral (M, H) dx
        # we need the scalar product of m and H

        # The first 3 index of m are the components (x,y,z) and the last index is the index of the cell
        #all_cells = np.sum(m[i, j, k, :])

        # define cell volume
        V = self._mesh.cell_volume
        eff_material = constants.mu_0 * self._Ms

        E = 0
        for i in m[3]:  
            E += (eff_material * np.dot(m[:, :, :, i], self._h))

        return - V * E
    