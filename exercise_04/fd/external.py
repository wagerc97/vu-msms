import numpy as np
from scipy import constants

class ExternalField(object):
    def __init__(self, mesh, material, h):
        """
        Initialize the external field with mesh, material properties, and external field.
        
        mesh: object with property 'n' = grid shape (Nx, Ny, Nz), and 'cell_volume' = float
        material: dict with 'Ms' (saturation magnetization in A/m)
        h: external field as a 3-element list or array [Hx, Hy, Hz] in Tesla
        """
        self._mesh = mesh
        self._Ms = material["Ms"]                    # [A/m]
        self._h = np.zeros(mesh.n + (3,))            # shape = (Nx, Ny, Nz, 3)
        self._h[:, :, :, :] = np.array(h)            # broadcast uniform external field [T]

    def h(self, t, m):
        """Return the field at time t and magnetization m — independent of t, m here."""
        return self._h

    def E(self, t, m):
        """
        Compute Zeeman energy: E = - mu_0 * Ms * ∑ (m · H) * V

        m: magnetization unit vector field, shape (Nx, Ny, Nz, 3)
        returns: Zeeman energy [J]
        """
        V = self._mesh.cell_volume                  # volume of a single cell [m³]
        mu0 = constants.mu_0                        # vacuum permeability [T·m/A]
        Ms = self._Ms                               # saturation magnetization [A/m]
        H = self._h                                 # external field [T]
        m_dot_H = np.sum(m * H, axis=3)             # dot product at each cell, result shape (Nx, Ny, Nz)

        E_total = -mu0 * Ms * np.sum(m_dot_H) * V   # total energy [J]
        return E_total
