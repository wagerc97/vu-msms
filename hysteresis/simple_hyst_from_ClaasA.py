import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# material constants
#mu0   = 4 * np.pi * 1e-7
#Ms    = 8e5            # Saturation magnetization [A/m]
#K     = 1e4            # Anisotropy constant [J/m^3]

def energy(phi, Hpar, Hperp):
    return np.sin(phi)**2 - Hpar*np.cos(phi) - Hperp*np.sin(phi)

#theta = np.radians(89.99)
theta = np.radians(45.)
Hmax = 2.2
Hs = np.concatenate((np.linspace(-Hmax, Hmax, 100), np.linspace(Hmax, -Hmax, 100)))
ms = []

phi = 0.
for H in Hs:
    # TODO make this harder
    Hpar  = H * np.cos(theta)
    Hperp = H * np.sin(theta)
    phi = minimize(energy, phi, args = (Hpar, Hperp)).x[0]
    ms.append(np.cos(theta-phi))

fig, ax = plt.subplots()
ax.plot(Hs, ms)

ax.set(xlabel='H', ylabel='M')
ax.grid()
plt.show()



if __name__ == '__main__':
    pass