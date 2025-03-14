# %% [markdown]
# # MuMag Standard Problem #4

# %% [markdown]
# ## Run Simulation

# %%
from magnumnp import *
import torch
import pathlib
from tqdm import tqdm

# I added this line to suppress the error
import torch._dynamo
torch._dynamo.config.suppress_errors = True

set_log_level(25) # show info_green, but hide info_blue
Timer.enable()
try:
    this_dir = pathlib.Path(__file__).resolve().parent
except:
    this_dir = pathlib.Path().resolve()

# initialize state
# We define a mesh
dt = 1e-11
n  = (100, 25, 1)
dx = (5e-9, 5e-9, 3e-9)
mesh = Mesh(n, dx)
state = State(mesh)

# We define the material
state.material = {
    "Ms": 8e5,
    "A": 1.3e-11,
    "alpha": 0.02
    }

# initialize field terms
# Which energy contributions
demag    = DemagField()
exchange = ExchangeField()
# H_ex as defined in the standard problem
external = ExternalField([-24.6e-3/constants.mu_0,
                          +4.3e-3/constants.mu_0,
                          0.0])

# initialize magnetization that relaxes into s-state
state.m = state.Constant([0,0,0])
state.m[1:-1,:,:,0]   = 1.0
state.m[(-1,0),:,:,1] = 1.0

# relax without external field
minimizer = MinimizerBB([demag, exchange])
minimizer.minimize(state)
state.write_vtk(state.m, "data/m0")

# perform integration with external field
# Now we add the external field
llg = LLGSolver([demag, exchange, external])
logger = Logger(this_dir / "data", ['t', 'm'], ['m'])

for i in tqdm(torch.arange(0, 1e-9, dt)):
    llg.step(state, dt)
    logger << state

Timer.print_report()


if __name__ == '__main__':
    print(f"\n\nDone")
