# load python modules
import os
import cupy as cp

xp = cp

from pyquda import init

# Create .cache directory for QUDA tuning parameters
if not os.path.exists(".cache"):
    os.makedirs(".cache", exist_ok=True)

Ls = 4
Lt = 8

init(
    [1, 1, 1, 1],
    enable_mps=True,
    grid_map="shared",
    backend="cupy",
    resource_path=".cache",
)

from pyquda_utils import core, io, source
from utils.bw_seq_pycontract import create_bw_seq_pycontract
from utils.bw_seq_pyquda import create_bw_seq_pyquda

width = 1.0
boost_in = boost_out = [0, 0, 1]
pf = [0, 0, 1, 0]
t_insert = 3
pol = ["PpUnpol"]
interpolation = "T5"

L = [Ls, Ls, Ls, Lt]
xi_0, nu = 2.464, 0.95
kappa = 0.115
mass = 1 / (2 * kappa) - 4
csw_r, csw_t = 0.91, 1.07
multigrid = None

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-12, 10000, xi_0, csw_r, csw_t, multigrid)
gauge = io.readQIOGauge("/home/jinchen/git/lat-software/PyQUDA_qTMD/tests/pycontract/weak_field.lime")

pos = [1,2,1,2]

# get forward propagator boosted source
srcD = source.propagator(latt_info, "point", pos)
dirac.loadGauge(gauge)
propag = core.invertPropagator(dirac, srcD, 1, 0) 

sequential_bw_prop_down_pycontract = create_bw_seq_pycontract(
    dirac,
    propag,
    pos,
    width,
    boost_out,
    pf,
    t_insert,
    pol,
    2,
    interpolation,
)

sequential_bw_prop_up_pycontract = create_bw_seq_pycontract(
    dirac,
    propag,
    pos,
    width,
    boost_in,
    pf,
    t_insert,
    pol,
    1,
    interpolation,
)

sequential_bw_prop_down_pyquda = create_bw_seq_pyquda(
    dirac,
    propag,
    pos,
    width,
    boost_out,
    pf,
    t_insert,
    pol,
    2,
    interpolation,
)

sequential_bw_prop_up_pyquda = create_bw_seq_pyquda(
    dirac,
    propag,
    pos,
    width,
    boost_in,
    pf,
    t_insert,
    pol,
    1,
    interpolation,
)


def print_compare_metrics(name, arr_a, arr_b, rtol=1e-8, atol=1e-10):
    a = cp.asarray(arr_a)
    b = cp.asarray(arr_b)
    diff = a - b
    abs_diff = cp.abs(diff)
    max_abs = float(cp.max(abs_diff).item())
    mean_abs = float(cp.mean(abs_diff).item())
    l2_diff = float(cp.linalg.norm(diff.ravel()).item())
    l2_ref = float(cp.linalg.norm(b.ravel()).item())
    rel_l2 = l2_diff / (l2_ref + 1e-300)
    is_close = bool(cp.allclose(a, b, rtol=rtol, atol=atol))

    print(f"[{name}]")
    print(f"shape={a.shape}, dtype={a.dtype}")
    print(f"max_abs={max_abs:.6e}")
    print(f"mean_abs={mean_abs:.6e}")
    print(f"rel_l2={rel_l2:.6e}")
    print(f"allclose(rtol={rtol:.0e}, atol={atol:.0e})={is_close}")
    print()


# Compare pycontract vs pyquda backward sequential propagators.
print_compare_metrics(
    "down: pycontract vs pyquda",
    sequential_bw_prop_down_pycontract,
    sequential_bw_prop_down_pyquda,
)
print_compare_metrics(
    "up: pycontract vs pyquda",
    sequential_bw_prop_up_pycontract,
    sequential_bw_prop_up_pyquda,
)