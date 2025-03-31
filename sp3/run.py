"""
Ziel: finde die Kantenlänge eine Kubus

"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from magnumnp import *

N = 10
Ms = 8e5
A = 13e-12


def calc(L, mag_state: str, folder: Path):
    # compute effective constants
    Keff = constants.mu_0 * Ms ** 2 / 2.
    l_ex = (A / Keff) ** 0.5

    l = L * l_ex  # sample size
    dx = l / N

    # cell size
    # initialize mesh
    n = (N, N, N)
    dx = (dx, dx, dx)
    origin = (-l / 2, -l / 2, -l / 2)
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
    llg = LLGSolver([aniso, demag, exchange])

    # relax vortex and compute energy
    state.m = state.Constant([0, 0, 0])
    x, y, z = mesh.SpatialCoordinate()

    if mag_state == "vortex":
        # <<< VORTEX STATE >>>
        # NOTE: Dieser state sieht wie eine Vortex aus. (Trick) Damit dreht sich die Magnetisierung überall herum.
        # Wir geben praktisch einen Tipp, wie das Energieminimum aussehen soll.
        state.m[:, :, :, 0] = l / 10  # X-ebene | Im Zentrum gewinnt diese kleine Komponente.
        state.m[:, :, :, 1] = -z  # y-ebene | Sonst überwiegen die y- und z-komponenten.
        state.m[:, :, :, 2] = y  # z-ebene

    elif mag_state == "flower":
        # <<< FLOWER STATE >>>
        state.m[:, :, :, 0] = 0  # X-ebene
        state.m[:, :, :, 1] = 0  # y-ebene
        state.m[:, :, :, 2] = 1  # z-ebene
    else:
        raise NotImplemented(f"unknown state: {mag_state}")

    normalize(state.m)  # normalize the vectors

    llg.relax(state)
    E_vortex = float(aniso.E(state) + demag.E(state) + exchange.E(state))

    # Log VTI information
    filename = folder / f"{mag_state}_{str(L)}.vti"
    write_vti(state.m, str(filename))
    return E_vortex



def main0(folder: Path):
    STATE = "flower"
    data = []  # List to store results
    for i in range(1):
        length = 8.0 + i / 10
        print(f"\n\nlength: {length}")
        e_vortex = calc(L=length, mag_state=STATE, folder=folder)
        data.append({"State": STATE, "L": length, "E_vortex": e_vortex})

    # Save DataFrame to CSV
    df = pd.DataFrame(data)
    csv_path = folder / "energy_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


# =============================================================
# Main Functions
# =============================================================

def main1(folder: Path):
    STATE = "flower"
    data = []  # List to store results
    for i in range(11):
        length = 8.0 + i / 10
        print(f"\n\nlength: {length}")
        e_vortex = calc(L=length, mag_state=STATE, folder=folder)
        data.append({"State": STATE, "L": length, "E_vortex": e_vortex})

    # Save DataFrame to CSV
    df = pd.DataFrame(data)
    csv_path = folder / "energy_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


def main2(folder: Path):
    data = []  # List to store results

    mag_states = ["vortex", "flower"]
    for mag_state in mag_states:
        for i in range(11):
            length = 8.0 + i / 10
            print(f"\n\nmag_state: {mag_state} and length: {length}")
            e_vortex = calc(L=length, mag_state=mag_state, folder=folder)
            data.append({"State": mag_state, "L": length, "E_vortex": e_vortex})

    # Save DataFrame to CSV
    df = pd.DataFrame(data)
    csv_path = folder / "energy_results.csv"
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
    plt.xlabel("L")
    plt.ylabel("E_vortex")
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
    FOLDER = Path(".") / "data"
    os.makedirs(FOLDER, exist_ok=True)

    #main0(FOLDER)
    #main1(FOLDER)
    main2(FOLDER)

    plot_energy_vs_length_for_states()


