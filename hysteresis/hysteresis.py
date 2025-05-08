import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# material constants
mu0 = 4 * np.pi * 1e-7
Ms = 8e5
K = 1e4


# energy function
def energy(phi, Hpar, Hperp):
    return K * np.sin(phi) ** 2 - mu0 * Ms * (Hpar * np.cos(phi) + Hperp * np.sin(phi))


# critical field Hc
Hc = 2 * K / (mu0 * Ms)
# angle of easy axis
theta = np.radians(90)  #> small tilt -> non perfect angle

#> Generate fields
#> External field's intensity changes
# EXPLANATION: 3x definitely above
fields = np.linspace(3*Hc, -3*Hc, 100)

magns = []
curr_min = 0
epsilon = 1e-6 #> perturbation to escape meta-stable point
#epsilon = 0

for H in fields:
    res = minimize(energy, curr_min+epsilon, args=(H * np.cos(theta), H * np.sin(theta)))
    phi_min = res.x[0]
    print(f"phi_min: {phi_min}")
    curr_min = phi_min
    magn = np.cos(phi_min-theta)
    magns.append(magn)

plt.plot(fields, magns)
plt.grid()
plt.xlabel("H")
plt.ylabel("M")
plt.title(f"Hysteresis loop")
plt.savefig(f"hysteresis_theta={theta}.png")
plt.show()
plt.close()

if __name__ == '__main__':
    ...