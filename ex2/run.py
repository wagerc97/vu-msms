"""
Ziel: finde die Kantenlänge eine Kubus

"""
import sys
from dis import show_code

from magnumnp import *
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FOLDER = Path(".") / "data"
os.makedirs(FOLDER, exist_ok=True)

N = 10          # number of mesh points
Ms = 8e5        # magnetisation
A = 13e-12      # ?
L = 1e9

# =============================================================
# main
# =============================================================

# compute effective constants
Keff = constants.mu_0 * Ms ** 2 / 2.  # Effective anisotropy constant
l_ex = (A / Keff) ** 0.5

l = L * l_ex  # sample size
dx = l / N

# cell size
# initialize mesh
n = (N, N, 1)  # cell size for z-coordinate is 1, x and z arbitrary large
dx = (dx, dx, dx)
origin = (-l / 2, -l / 2, -l / 2) # of coordinate system
mesh = Mesh(n, dx, origin)
state = State(mesh)
state.material = {
    "Ms": Ms,    # Saturation magnetization
    "A": A, # Exchange stiffness
    "Ku": 0.1 * Keff,   # Uniaxial anisotropy constant
    "Ku_axis": [0, 0, 1],
    "alpha": 1.0}

# initialize field terms and LLG solver
#aniso = UniaxialAnisotropyField()
demag = DemagField()
#exchange = ExchangeField()

# This time: Relaxation is unnecessary, because our Ms is homogenous
#llg = LLGSolver([aniso, demag, exchange])

# relax vortex and compute energy
state.m = state.Constant([0, 0, 0])
x, y, z = mesh.SpatialCoordinate()

# Start with angle in y = 1
# kipp den winkel über die x Achse, dabei bleibt y immer gleich 0
# loop over angles

# Define the range of angles to loop over
angles = np.linspace(0, np.pi, num=50)  # 100 angles from 0 to π

# Set the initial condition: vector pointing in z-direction
state.m[:, :, :, 0] = 0  # X-component of the vector
state.m[:, :, :, 1] = 0  # Y-component of the vector
state.m[:, :, :, 2] = 1  # Z-component of the vector

# To collect values
data = []  # List to store results


def get_analytical_energy():
    """
    Calculate the energy analytically.

    Equation for idealized demagnetization field:

    E_demag = - 0.5 * V * mu_0 * M_dash * N_tilde * M_vec

    where
        - V is the volume of the magnetic shape.
        - M_dash and M_vec represent magnetization properties.
        - N_tilde is the demagnetization tensor, which depends on the shape.
        - mu_0 is the permeability of free space.

    """

    #print(f"state.m of shape {state.m.shape}: {state.m}")  #> torch.Size([10, 10, 1, 3])

    V = dx[2]  # The film is infinitely large, so the volume is typically taken per unit area
               # (thickness d is relevant).
    print(f"V1: {V}")
    V = dx[0] * dx[1] * dx[0]
    print(f"V2: {V}")
    # Demagnetization factor (thin film approximation)
    N_xx, N_yy, N_zz = n  # Thin film: N_zz dominates

    # Compute magnetization components as a function of the angle
    M_x = Ms * np.sin(angle)
    M_y = 0
    M_z = Ms * np.cos(angle)

    # each axis contribution individually
    E_demag = -0.5 * V * constants.mu_0 * (N_xx * M_x**2 + N_yy * M_y**2 + N_zz * M_z**2)

    return E_demag


# Loop over angles and update the x and z components
# INTERPRETATION:
#   Starting at θ = 0° (0 radians) → Vector (0, 0, 1) (pointing up along z-axis).
#   Ending at θ = 180° (π radians) → Vector (0, 0, -1) (pointing down along z-axis).
#   This means the vector flips completely from +z to -z.
for angle in angles:
    state.m[:, :, :, 0] = np.sin(angle)  # X-component: rotated around x-axis
    #state.m[:, :, :, 1] = 0  # Y-component remains 0 as per the requirement
    state.m[:, :, :, 2] = np.cos(angle)  # Z-component: remains on z-plane

    normalize(state.m)  # normalize the vectors

    # we dont need to relax, as M is homogenous
    #llg.relax(state)

    E_sim = float(demag.E(state))  # get energy from demagnetization field
    E_calc = get_analytical_energy()
    data.append({"angle": angle, "E_sim": E_sim, "E_calc": E_calc})  # collect values

    # Log VTI information
    #filename = FOLDER / f"{str(L)}.vti"
    #write_vti(state.m, str(filename))


# Save DataFrame to CSV
df = pd.DataFrame(data)
csv_path = FOLDER / "demag_energy.csv"
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")





def plot():
    """
    Compare the energy (E_vortex) across the two states over the values of L.
    :return:
    """
    # Load the CSV file
    csv_path = FOLDER / "demag_energy.csv"
    df = pd.read_csv(csv_path)
    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(df["angle"], df["E_sim"], marker='o', label="E_sim")
    plt.plot(df["angle"], df["E_calc"], marker='s', label="E_calc")
    # Labels and title
    plt.xlabel(r"angle of magnetization $\phi$")
    plt.ylabel(r"E ($K_m$)")
    aspect_ratio = 1 / L
    plt.title(
        r"Development of energy as angle of magnetization $\phi$ flips" + "\n" +
        r"$M_z$ by 180° from z+ to z-. " + f"Aspect ratio: {aspect_ratio}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = FOLDER / "plot"
    plt.savefig(plot_path)
    # Show the plot
    plt.show()
    plt.close()


if __name__ == '__main__':
    plot()

