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
        self._mesh: np.ndarray = mesh                   # mesh object  
        self._Ms: float = material["Ms"]                # [A/m]
        self._h: np.ndarray = np.zeros(mesh.n + (3,))   # shape = (Nx, Ny, Nz, 3)
        self._h[:, :, :, :] = np.array(h)               # broadcast uniform external field [T]


    def h(self, t, m) -> np.ndarray:
        """Return the field at time t and magnetization m — independent of t, m here."""
        print(f"self._h.shape: {self._h.shape}")  #> (100, 25, 1, 3)
        return self._h


    def E(self, t, m: np.ndarray) -> float:
        """
        Compute Zeeman energy: 
        E = -V * mu_0 * Ms * sum( m · H )

        Note: For a discretized mesh, we sum over all cells:

        returns: Zeeman energy [J]
        """
        V: np.ndarray = self._mesh.cell_volume      # volume of a single cell [m³]
        mu0 = constants.mu_0                        # vacuum permeability [T·m/A]
        Ms: float = self._Ms                        # saturation magnetization [A/m]
        H: np.ndarray = self._h                     # external field [T]

        # This line computes the dot product between the magnetization vector field m 
        #  and the external magnetic field H, at each grid point in the 3D mesh.
        # ~ the per-cell dot products
        #
        # Why do we use axis=3?
        # - Because m is a 4D array with shape (Nx, Ny, Nz, 3), and we want to sum over the last dimension (3).
        # - The last dimension (axis=3) corresponds to the 3 components of the magnetization vector (x,y,z).
        m_dot_H: np.ndarray = np.sum(m * H, axis=3)  # has shape (Nx, Ny, Nz)

        # Sum over each cell's dot product
        E_total: float = - V * mu0 * Ms * np.sum(m_dot_H)  # total energy [J]
        print(f"E_total: {E_total}")  #> E_total: -1e-24

        return E_total
