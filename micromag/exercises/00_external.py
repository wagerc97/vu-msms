from fd import *
import numpy as np
from pathlib import Path

# initialize mesh
n  = (100, 25, 1)
dx = (5e-9, 5e-9, 3e-9)
mesh = Mesh(n, dx)

# initialize material
material = {
        "Ms": 8e5,
        "A": 1.3e-11,
        "K": 1e4,
        "K_axis": (0,0,1)  # anisotropy axis points in z-direction
        }

# initialize external field pointing in y-direction
external = ExternalField(mesh, material, (0,1,0))  # apply external field in y-direction

# initialize magnetization pointing in x=direction
m0 = np.zeros(n + (3,))
m0[:,:,:,0] = 1.0

# minimize energy
result_path = Path(__file__).parents[1] / "output" / "external/"
print(f"Results folder: {result_path}")
result_path.mkdir(parents=True, exist_ok=True)

write_vtr(m0, str(result_path/"m_start"), mesh)
print("wrote initial state to filepath: ", result_path/"m_start")
print("Minimizing energy...")
minimizer = Minimizer([external])
m = minimizer.minimize(m0, verbose=True)
write_vtr(m, str(result_path/"m_relaxed"), mesh)
print("wrote relaxed state to filepath: ", result_path/"m_relaxed")
