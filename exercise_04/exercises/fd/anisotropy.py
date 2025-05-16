import numpy as np
from scipy import constants

class AnisotropyField(object):
    def __init__(self, mesh, material):
        self._mesh = mesh
        self._Ms = material["Ms"]
        self._K = material["K"]
        self._K_axis = np.array(material["K_axis"]).reshape(1,1,1,-3)


    def h(self, t, m) -> np.ndarray:
        """Compute the anisotropy field.
        t (float): Time.
        m (ndarray): Magnetization vector.
        """
        # TODO
        # Implement anisotropy field:
        #raise NotImplementedError
        # 2 * K / (mu_0 * Ms) * K_axis * <K_axis, m>
        #
        mu_0 = constants.mu_0
        Ms = self._Ms
        K = self._K
        e = self._K_axis
        #print(f"e.shape: {e.shape}")  #> (1, 1, 1, 3)
        #print(f"m.shape: {m.shape}")  #> (100, 25, 1, 3)

        # Compute the dot product between the anisotropy axis e and the magnetization vector field m
        m_dot_e = np.sum(m * e, axis=3, keepdims=True)  # # shape: (Nx, Ny, Nz, 1)
        
        # Compute the anisotropy field
        h = ((2 * K) / (mu_0 * Ms)) * m_dot_e * e  # broadcast to shape (Nx, Ny, Nz, 3)
        #print(f"h.shape: {h.shape}")  #> (100, 25, 1, 3)
        return h


    def E(self, t, m) -> np.ndarray:
        """Compute the anisotropy energy. 

        NOTE: 
            - For E we do not divide by mu_0*Ms, because we only do that for the field.
            - The K should be negative. 

        t (float): Time.
        m (ndarray): Magnetization vector.
        """
        # TODO
        # Implement anisotropy energy:
        #raise NotImplementedError
        ## Given 
        # K / (mu_0 * Ms) * <K_axis, m>^2
        ## Corrected ??? 
        # -K * <K_axis, m>^2
        #
        K = self._K
        e = self._K_axis
        #print(f"e.shape: {e.shape}")  #> (1, 1, 1, 3)
        #print(f"m.shape: {m.shape}")  #> (100, 25, 1, 3)

        # Compute the dot product between the magnetization vector field m and the anisotropy axis e
        e_dot_m = np.sum(e * m, axis=3)  # shape: (Nx, Ny, Nz)

        # Compute the anisotropy energy
        E = -K * e_dot_m**2
        #print(f"E.shape: {E.shape}")
        return E
    