import numpy as np
import matplotlib.pyplot as plt

# material constants
mu0   = 4 * np.pi * 1e-7
Ms    = 8e5   # Saturation magnetization [A/m]
K     = 1e4   # Anisotropy constant [J/m^3]

def energy(phi, Hpar, Hperp):
    # TODO implement (use np.sin and np.cos)

    ...

angles   = np.linspace(-90, 270, 361)
energies = []

for i in angles:
    energies.append(energy(np.radians(i), 10e-3/mu0, 0))

fig, ax = plt.subplots()
ax.plot(angles, energies)

ax.set(xlabel='angle', ylabel='e')
ax.grid()
plt.show()