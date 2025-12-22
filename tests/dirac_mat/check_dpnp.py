import os
import numpy as np
from pyquda_utils import core, io, source

# Create .cache directory for QUDA tuning parameters
import shutil
if core.getMPIRank() == 0:
    if os.path.exists(".cache"):
        shutil.rmtree(".cache")
    os.makedirs(".cache", exist_ok=True)

import sys
if len(sys.argv) > 1:
    tag = sys.argv[1]
else:
    raise ValueError("Please provide a tag")

Ls = 8
Lt = 8
xi_0, nu = 1.0, 1.0
mass = -0.038888  # kappa = 0.12623
csw_r = 1.02868
csw_t = 1.02868
multigrid = None

core.init(None, [Ls, Ls, Ls, Lt], enable_mps=True, grid_map="shared", backend="dpnp", backend_target="sycl", resource_path=".cache")
latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)

dirac = core.getClover(latt_info, mass, 1e-10, 10000, xi_0, csw_r, csw_t, multigrid)
# dirac = core.getWilson(latt_info, mass, 1e-10, 10000, multigrid)
gauge = io.readNERSCGauge(f"../../test_gauge/S8T8_wilson_b6.0")
dirac.loadGauge(gauge)

pos = [1, 2, 3, 4]
noise = source.point(latt_info, pos, 0, 0)
# if core.getMPISize() == 1 and tag == "S8T8_local":
#     noise = core.LatticeFermion(latt_info)
#     np.random.seed(42)  # seed
#     noise.data = np.random.normal(0, 2**-0.5, noise.shape) + 1j * np.random.normal(0, 2**-0.5, noise.shape)
#     noise.save(f"output/rand_noise.npy")
# else:
#     noise = core.LatticeFermion.load(f"output/rand_noise.npy")
noise.toDevice()

core.getLogger().info("")
core.getLogger().info("TESTING: dirac.mat(src_point)")
dirac.invert_param.verbosity = 2
dirac.setPrecision(sloppy=8)
dirac.mat(noise).save(f"output/{tag}_mat_noise.npy")
dirac.invert(noise).save(f"output/{tag}_inv_noise.npy")
core.getLogger().info("TESTING: dirac.mat(src_point) DONE")
core.getLogger().info("")