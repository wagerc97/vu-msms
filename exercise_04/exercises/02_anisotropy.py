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
        "K": 1e5,
        "K_axis": (0,0,1)
        }

# initialize anisotropy field
aniso = AnisotropyField(mesh, material)

# initialize random magnetization
m0 = np.random.rand(n[0], n[1], n[2], 3) - 0.5
m0 = m0 / np.linalg.norm(m0, axis=3).repeat(3).reshape(m0.shape)

# minimize energy
result_path = Path("../output/anisotropy/").resolve()
print(f"Results folder: {result_path}")
result_path.mkdir(parents=True, exist_ok=True)

write_vtr(m0, str(result_path/"m_start"), mesh)
print("wrote initial state to filepath: ", result_path/"m_start")
print("Minimizing energy...")
minimizer = Minimizer([aniso])
m = minimizer.minimize(m0, 1e-4, 1e-4)
write_vtr(m, str(result_path/"m_relaxed"), mesh)
print("wrote relaxed state to filepath: ", result_path/"m_relaxed")