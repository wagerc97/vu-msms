"""
Ziel: finde die Kantenlänge eine Kubus

"""
from magnumnp import *
import numpy as np

N = 10
Ms = 8e5
A = 13e-12


def calc(L, mag_state: str):
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

    write_vti(state.m, f"vortex_{str(L)}_{mag_state}.vti")
    return E_vortex


def main1():
    STATE = "flower"
    with open(f"energy.txt", 'w') as file:
        for i in range(11):
            lenght = 8.0 + i / 10
            print(f"\n\nlenght: {lenght}")
            e_vortex = calc(L=lenght, mag_state=STATE)
            print(e_vortex)

            line = f"state: {STATE} | L: {str(lenght)} -> e_vortex: {e_vortex}\n"
            file.write(line)


def main2():
    with open(f"energy.txt", 'w') as file:
        states = ["vortex", "flower"]
        for state in states:
            for i in range(11):
                lenght = 8.0 + i / 10
                print(f"\n\nlenght: {lenght}")
                e_vortex = calc(L=lenght, mag_state=state)
                print(e_vortex)

                line = f"state: {state} | L: {str(lenght)} -> e_vortex: {e_vortex}\n"
                file.write(line)


if __name__ == '__main__':
    main1()
