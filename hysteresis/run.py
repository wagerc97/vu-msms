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

# prepare arrays for field angles and critical fields
#> EXPLANATION: only one quadrant
angles = np.radians(np.linspace(-90, 90, 181))
Hpars, Hperps = [], []

# prepare plot
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.set(xlabel='Hpar', ylabel='Hperp')
ax.set_aspect('equal')
ax.set_xlim(-Hc, Hc)
ax.set_ylim(-Hc, Hc)
#> add title
plt.title(r"Complete Hysteresis Curve for field angle $\gamma$")



def abc():

    # TODO plot analytical Stoner Wohlfarth asteroid
    ax.plot(..., ...)  #> We plot perpendicular vs parallel field

    ### TODO comment if plot exercise is done to move on to next task
    plt.ioff()
    plt.show()

    print(f"plot is ready ")
    #exit()
abc()


# add plot with empty values for Hpar and Hperp
sw, = ax.plot(Hpars, Hperps)

for alpha in angles:
    print(alpha)
    phi = np.pi  #> what is this

    # TODO find critical field H for angle alpha
    Hpar  = -Hc * np.cos(alpha) ** 3
    Hperp = Hc * np.sin(alpha) ** 3
    # Store values of critical field in variables "Hpar" and "Hperp"
    #
    # tip: use minimize(xxx, method = "L-BFGS-B")
    mini = minimize(energy, 100, args=(phi, Hpar, Hperp))
    print(f"mini: {mini}")

    # Append critical field components and update plot
    Hpars.append(Hpar)
    Hperps.append(Hperp)

    sw.set_xdata(Hpars)
    sw.set_ydata(Hperps)
    fig.canvas.draw()

plt.ioff()
plt.show()