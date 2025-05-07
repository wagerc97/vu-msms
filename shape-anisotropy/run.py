"""
Ziel: finde die Kantenlänge eine Kubus

"""

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
Keff = constants.mu_0 * Ms ** 2 / 2.
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
    "Ms": Ms,
    "A": A,
    "Ku": 0.1 * Keff,
    "Ku_axis": [0, 0, 1],
    "alpha": 1.0}

# initialize field terms and LLG solver
aniso = UniaxialAnisotropyField()
demag = DemagField()
exchange = ExchangeField()

# This time: Relaxation is unnecessary, because our Ms is homogenous
#llg = LLGSolver([aniso, demag, exchange])

# relax vortex and compute energy
state.m = state.Constant([0, 0, 0])
x, y, z = mesh.SpatialCoordinate()

# Start with angle in y = 1
# kipp den winkel über die x Achse, dabei bleibt y immer gleich 0
# loop over angles

# Define the range of angles to loop over
angles = np.linspace(0, np.pi, num=100)  # 100 angles from 0 to π

# Set the initial condition: vector pointing in z-direction
state.m[:, :, :, 0] = 0  # X-component of the vector
state.m[:, :, :, 1] = 0  # Y-component of the vector
state.m[:, :, :, 2] = 1  # Z-component of the vector

# To collect values
data = []  # List to store results


# Loop over angles and update the x and z components
for angle in angles:
    state.m[:, :, :, 0] = np.sin(angle)  # X-component: rotated around x-axis
    #state.m[:, :, :, 1] = 0  # Y-component remains 0 as per the requirement
    state.m[:, :, :, 2] = np.cos(angle)  # Z-component: remains on z-plane

    normalize(state.m)  # normalize the vectors

    # we dont need to relax, as M is homogenous
    #llg.relax(state)

    E = float(demag.E(state))  # get energy from demagnetization field
    data.append({"angle": angle, "E": E})  # collect values

    # Log VTI information
    #filename = FOLDER / f"{str(L)}.vti"
    #write_vti(state.m, str(filename))


# Save DataFrame to CSV
df = pd.DataFrame(data)
csv_path = FOLDER / "demag_energy.csv"
df.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")





def plot_energy_vs_length_for_states():
    """
    Compare the energy (E_vortex) across the two states over the values of L.
    :return:
    """
    # Load the CSV file
    csv_path = FOLDER / "energy_results.csv"
    df = pd.read_csv(csv_path)
    # Filter the data based on state
    df_vortex = df[df["State"] == "vortex"]
    df_flower = df[df["State"] == "flower"]
    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(df_vortex["L"], df_vortex["E_vortex"], marker='o', linestyle='-', label="Vortex", color='blue')
    plt.plot(df_flower["L"], df_flower["E_vortex"], marker='s', linestyle='--', label="Flower", color='red')
    # Labels and title
    plt.xlabel(r"L ($l_{ex}$)")
    plt.ylabel(r"E_vortex ($K_m$)")
    plt.title("Comparison of E_vortex for Different States")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = FOLDER / "plot"
    plt.savefig(plot_path)
    # Show the plot
    plt.show()
    plt.close()


if __name__ == '__main__':
    ...

