# load python modules
import os
from typing import Sequence
from math import pi

import h5py
import numpy as np

from pyquda import init
from pyquda.field import Ns, Nc
from pyquda.field import LatticeInfo, LatticeFermion, LatticePropagator, LatticeComplex
from pyquda_comm import getMPIRank, getCoordFromRank
from pyquda_utils import core, gamma, io, source
from pyquda_utils.phase import MomentumPhase
from pyquda_utils.fft import fft, ifft

# Create .cache directory for QUDA tuning parameters
if not os.path.exists(".cache"):
    os.makedirs(".cache", exist_ok=True)

# --------------------------
# Helper functions
# --------------------------

def _get_xp_from_array(a):
    """Return the base module of the array's type, e.g. cupy / numpy."""
    if a is None:
        return np
    base = type(a).__module__.split('.')[0]
    return __import__(base)


def _ensure_backend(x, xp):
    """Move x to the same backend as xp if needed."""
    if type(x).__module__.split('.')[0] == xp.__name__:
        return x
    if hasattr(xp, "asarray"):
        return xp.asarray(x)
    return xp.array(x)


def _asarray_on_queue(val, xp, ref_arr):
    """Creates an array 'val' on the same backend and SYCL queue as 'ref_arr'."""
    if xp.__name__ == 'dpnp' and hasattr(ref_arr, 'sycl_queue'):
        return xp.asarray(val, sycl_queue=ref_arr.sycl_queue)
    return xp.asarray(val)


def mpi_print(latt_info, message):
    if latt_info.mpi_rank == 0:
        print(message)


def srcLoc_distri_eq(L, src_origin):
    source_positions = []
    div = 4
    for i in range(div):
        for j in range(div):
            for k in range(div):
                for l in range(div):
                    source_positions += [[
                        round(i * L[0] / div + src_origin[0]) % L[0],
                        round(j * L[1] / div + src_origin[1]) % L[1],
                        round(k * L[2] / div + src_origin[2]) % L[2],
                        round(l * L[3] / div + src_origin[3]) % L[3]
                    ]]
    return source_positions


# --------------------------
# IO functions
# --------------------------

def get_c2pt_file_tag(data_dir, lat, cfg, ama, src, sm):
    cfg_tag = str(cfg)
    lat_tag = str(lat) + ".c2pt"
    ama_tag = str(ama)
    src_tag = "x" + str(src[0]) + "y" + str(src[1]) + "z" + str(src[2]) + "t" + str(src[3])
    sm_tag = str(sm)
    return data_dir + "/c2pt/" + lat_tag + "." + cfg_tag + "." + ama_tag + "." + src_tag + "." + sm_tag


def save_proton_c2pt_hdf5(corr, tag, gammalist, plist):
    roll = -int(tag.split(".")[4].split('t')[1])
    save_h5 = tag + ".h5"
    f = h5py.File(save_h5, 'w')
    sm = f.create_group("SS")
    for ig, gm in enumerate(gammalist):
        g = sm.create_group(gm)
        for ip, p in enumerate(plist):
            dataset_tag = "PX" + str(p[0]) + "PY" + str(p[1]) + "PZ" + str(p[2])
            g.create_dataset(dataset_tag, data=np.roll(corr[ig][ip], roll, axis=0))
    f.close()


# --------------------------
# Boosted smearing
# --------------------------

def _exp_complex(xp, real, imag):
    if xp.__name__ == "torch":
        return xp.exp(real) * (xp.cos(imag) + 1j * xp.sin(imag))
    return xp.exp(real + 1j * imag)


def _get_global_grid_coords(xp, latt_info: LatticeInfo):
    """Generate the global coordinates of the MPI Rank."""
    Lx, Ly, Lz, Lt = latt_info.size
    Gx, Gy, Gz, Gt = latt_info.global_size

    rank = getMPIRank()
    coords = getCoordFromRank(rank)

    off_t = coords[3] * Lt
    off_z = coords[2] * Lz
    off_y = coords[1] * Ly
    off_x = coords[0] * Lx

    rx_local = xp.arange(Lx, dtype=xp.float64)
    ry_local = xp.arange(Ly, dtype=xp.float64)
    rz_local = xp.arange(Lz, dtype=xp.float64)

    rx = (rx_local + off_x + Gx / 2) % Gx - Gx / 2
    ry = (ry_local + off_y + Gy / 2) % Gy - Gy / 2
    rz = (rz_local + off_z + Gz / 2) % Gz - Gz / 2

    return rx, ry, rz


def _build_kernel_realspace_distributed(xp, latt_info: LatticeInfo, w: float, boost: Sequence[float]):
    """Build the distributed real space Gaussian kernel."""
    rx, ry, rz = _get_global_grid_coords(xp, latt_info)
    Lx, Ly, Lz, Lt = latt_info.size
    Gx, Gy, Gz, Gt = latt_info.global_size

    kx, ky, kz = boost

    rx = rx[None, None, :]
    ry = ry[None, :, None]
    rz = rz[:, None, None]

    real = (-0.5 / (w * w)) * (rx ** 2 + ry ** 2 + rz ** 2)
    imag = 2 * pi * ((kx / Gx) * rx + (ky / Gy) * ry + (kz / Gz) * rz)

    k_xyz = _exp_complex(xp, real, imag)

    kernel_field = LatticeComplex(latt_info)
    k_full_local = xp.zeros((Lt, Lz, Ly, Lx), dtype=xp.complex128)
    k_full_local[:] = k_xyz[None, ...]

    if xp.__name__ == "numpy":
        k_full_local_cpu = k_full_local
    else:
        k_full_local_cpu = xp.asnumpy(k_full_local)
    cb_data = latt_info.evenodd(k_full_local_cpu, False)

    kernel_field.data = _ensure_backend(cb_data, xp)

    return kernel_field


def _boosted_smearing_fermion(src: LatticeFermion, *, w: float, boost: Sequence[float]):
    """Core implementation of boosted smearing for a single fermion."""
    latt_info: LatticeInfo = src.latt_info
    xp = _get_xp_from_array(src.data)

    psi_p = fft(src, fft3d=True, backend="cupy" if xp.__name__ == "cupy" else "numpy")

    K_xyz = _build_kernel_realspace_distributed(xp, latt_info, w, boost)
    K_p = fft(K_xyz, fft3d=True, backend="cupy" if xp.__name__ == "cupy" else "numpy")

    psi_p.data = psi_p.data * K_p.data[..., None, None]

    psi_smeared = ifft(psi_p, fft3d=True, backend="cupy" if xp.__name__ == "cupy" else "numpy")

    return psi_smeared


def boosted_smearing(src, *, w: float, boost: Sequence[float]):
    if isinstance(src, LatticeFermion):
        return _boosted_smearing_fermion(src, w=w, boost=boost)
    if isinstance(src, LatticePropagator):
        out = LatticePropagator(src.latt_info)
        for s in range(Ns):
            for c in range(Nc):
                f_sm = _boosted_smearing_fermion(src.getFermion(s, c), w=w, boost=boost)
                out.setFermion(f_sm, s, c)
        return out
    raise TypeError(f"boosted_smearing: unsupported src type: {type(src)}")


# --------------------------
# Proton TMD measurement
# --------------------------

# Gamma matrices
my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
my_pyquda_gammas = [
    gamma.gamma(15), gamma.gamma(8), gamma.gamma(7), gamma.gamma(1),
    gamma.gamma(14), gamma.gamma(2), gamma.gamma(13), gamma.gamma(4),
    gamma.gamma(11), gamma.gamma(0), gamma.gamma(9), gamma.gamma(3),
    gamma.gamma(5), gamma.gamma(10), gamma.gamma(6), gamma.gamma(12)
]

Cg5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(15)
CgT5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(7)
CgZ5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(11)


class proton_TMD:
    def __init__(self, parameters):
        self.width = parameters["width"]
        self.pos_boost = parameters["boost_in"]
        self.pilist = parameters["p_2pt"]

    def contract_2pt_TMD(self, latt_info, prop_f, phases, tag, interpolator="5"):
        if interpolator == "5":
            gamma_insert = Cg5
        elif interpolator == "T5":
            gamma_insert = CgT5
        elif interpolator == "Z5":
            gamma_insert = CgZ5
        else:
            raise ValueError(f"Invalid interpolator: {interpolator}")

        mpi_print(latt_info, "Begin sink smearing")
        prop_f = boosted_smearing(prop_f, w=self.width, boost=self.pos_boost)
        mpi_print(latt_info, "Sink smearing completed")

        xp = _get_xp_from_array(prop_f.data)
        P_2pt_gamma_host = xp.zeros((16, latt_info.Lt, 4, 4), dtype=prop_f.data.dtype)
        P_2pt_gamma = _asarray_on_queue(P_2pt_gamma_host, xp, prop_f.data)

        for gamma_idx, gamma_pyq_host in enumerate(my_pyquda_gammas):
            gamma_device = _asarray_on_queue(gamma_pyq_host, xp, prop_f.data)
            P_2pt_local = _asarray_on_queue(xp.zeros((latt_info.Lt, 4, 4), dtype=prop_f.data.dtype), xp, prop_f.data)
            P_2pt_local[:] = gamma_device
            P_2pt_gamma[gamma_idx] = P_2pt_local

        epsilon_host = xp.zeros((3, 3, 3), dtype=prop_f.data.real.dtype)
        for a in range(3):
            b = (a + 1) % 3
            c = (a + 2) % 3
            epsilon_host[a, b, c] = 1
            epsilon_host[a, c, b] = -1
        epsilon = _asarray_on_queue(epsilon_host, xp, prop_f.data)

        phases = _asarray_on_queue(phases, xp, prop_f.data)
        gamma_insert = _asarray_on_queue(gamma_insert, xp, prop_f.data)

        # --- Term 1 ---
        term1_sink = xp.einsum(
            "abc, ij, wtzyxikad, wtzyxjlbe -> wtzyxklcde",
            epsilon, gamma_insert, prop_f.data, prop_f.data,
            optimize=True
        )
        term1_p3 = xp.einsum(
            "gtmn, wtzyxmncf -> gwtzyxcf",
            P_2pt_gamma, prop_f.data,
            optimize=True
        )
        term1 = xp.einsum(
            "def, pwtzyx, kl, wtzyxklcde, gwtzyxcf -> gpt",
            epsilon, phases, gamma_insert, term1_sink, term1_p3,
            optimize=True
        )
        del term1_sink, term1_p3

        # --- Term 2 ---
        term2_sink = xp.einsum(
            "abc, ij, wtzyxikad, wtzyxjnbe -> wtzyxkncde",
            epsilon, gamma_insert, prop_f.data, prop_f.data,
            optimize=True
        )
        term2_p3 = xp.einsum(
            "gtmn, wtzyxmlcf -> gwtzyxnlcf",
            P_2pt_gamma, prop_f.data,
            optimize=True
        )
        term2 = xp.einsum(
            "def, pwtzyx, kl, wtzyxkncde, gwtzyxnlcf -> gpt",
            epsilon, phases, gamma_insert, term2_sink, term2_p3,
            optimize=True
        )
        del term2_sink, term2_p3

        corr = -term1 - term2

        if xp.__name__ == "numpy":
            corr_collect = core.gatherLattice(corr, [2, -1, -1, -1])
        else:
            corr_collect = core.gatherLattice(xp.asnumpy(corr), [2, -1, -1, -1])

        if latt_info.mpi_rank == 0:
            save_proton_c2pt_hdf5(corr_collect, tag, my_gammas, self.pilist)
        del corr, corr_collect


# ==========================
# Main script
# ==========================

# Global parameters
data_dir = "tests/pyq_vs_gpt/data"
lat_tag = "S8T8_pyquda_local"
interpolation = "T5"
sm_tag = "1HYP_GSRC_W90_k3_" + interpolation
Ls = 8
Lt = 8
conf = 0

# --------------------------
# initiate quda
# --------------------------
init(None, [Ls, Ls, Ls, Lt], enable_mps=True, grid_map="shared", backend="cupy", resource_path=".cache")

# --------------------------
# Setup parameters
# --------------------------
parameters = {
    "eta": [0],
    "b_z": 2,
    "b_T": 2,
    "qext": [[x, y, z, 0] for x in [2] for y in [-2] for z in [0]],
    "pf": [0, 0, 9, 0],
    "p_2pt": [[x, y, z, 0] for x in [2] for y in [-2] for z in [0]],
    "boost_in": [0, 0, 3],
    "boost_out": [0, 0, 3],
    "width": 9.0,
    "pol": ["PpUnpol"],
    "t_insert": 4,
    "save_propagators": False,
}
Measurement = proton_TMD(parameters)

# --------------------------
# Load gauge and create inverter
# --------------------------

L = [Ls, Ls, Ls, Lt]
xi_0, nu = 1.0, 1.0
mass = -0.038888
csw_r = 1.02868
csw_t = 1.02868

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-10, 10000, xi_0, csw_r, csw_t, None)
dirac.setPrecision(sloppy=8)
gauge = io.readNERSCGauge(f"/home/jinchen/git/lat-software/PyQUDA_qTMD/test_gauge/S8T8_wilson_b6.0")

# Setup source positions
src_shift = np.array([7, 11, 13, 23])
src_origin = np.array([int(conf) % L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin)

src_shift = np.array([15, 19, 21, 31])
src_origin = np.array([int(conf) % L[i] for i in range(4)]) + src_shift
src_positions = src_positions + srcLoc_distri_eq(L, src_origin)

src_shift = np.array([15, 19, 21, 27])
src_origin = np.array([int(conf) % L[i] for i in range(4)]) + src_shift
src_positions = src_positions + srcLoc_distri_eq(L, src_origin)

src_shift = np.array([15, 19, 17, 27])
src_origin = np.array([int(conf) % L[i] for i in range(4)]) + src_shift
src_positions = src_positions + srcLoc_distri_eq(L, src_origin)

src_production = src_positions[0:1]

# --------------------------
# Start measurements
# --------------------------

for ipos, pos in enumerate(src_production):
    srcD = source.propagator(latt_info, "point", pos)
    srcDp = boosted_smearing(srcD, w=parameters["width"], boost=parameters["boost_in"])
    dirac.loadGauge(gauge)
    propag = core.invertPropagator(dirac, srcDp, 1, 0)

    # Contract 2pt TMD
    tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    p_2pt_xyz = [[-v[0], -v[1], -v[2]] for v in parameters["p_2pt"]]
    phases_2pt = MomentumPhase(latt_info).getPhases(p_2pt_xyz, x0=pos)

    Measurement.contract_2pt_TMD(latt_info, propag, phases_2pt, tag, interpolation)
