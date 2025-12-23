# load python modules
import os
import h5py
import numpy as np

# load gpt modules
import gpt as g

# load pyquda modules
from pyquda import init
from pyquda_utils import core, gpt

# Create .cache directory
if not os.path.exists(".cache"):
    os.makedirs(".cache", exist_ok=True)

# --------------------------
# Helper functions
# --------------------------

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
        grp = sm.create_group(gm)
        for ip, p in enumerate(plist):
            dataset_tag = "PX" + str(p[0]) + "PY" + str(p[1]) + "PZ" + str(p[2])
            grp.create_dataset(dataset_tag, data=np.roll(corr[ig][ip], roll, axis=0))
    f.close()


# --------------------------
# Gamma matrices and projections
# --------------------------

my_gammas = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]

# Projection matrices
Cg5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma[5].tensor()
CgT5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma["T"].tensor() * g.gamma[5].tensor()
CgZ5 = (1j * g.gamma[1].tensor() * g.gamma[3].tensor()) * g.gamma["Z"].tensor() * g.gamma[5].tensor()


# --------------------------
# Proton TMD measurement class
# --------------------------

class proton_TMD:
    def __init__(self, parameters):
        self.pilist = parameters["p_2pt"]
        self.width = parameters["width"]
        self.pos_boost = parameters["boost_in"]

    def create_src_2pt(self, pos, trafo, grid):
        """Create boosted, smeared source."""
        srcD = g.mspincolor(grid)
        g.create.point(srcD, pos)
        srcDp = g.create.smear.boosted_smearing(trafo, srcD, w=self.width, boost=self.pos_boost)
        return srcDp

    def make_mom_phases_2pt(self, grid, origin=None):
        """Make list of complex phases for momentum projection."""
        one = g.identity(g.complex(grid))
        pp = [-2 * np.pi * np.array(pi) / grid.fdimensions for pi in self.pilist]
        P = g.exp_ixp(pp, origin)
        mom = [g.eval(pp * one) for pp in P]
        return mom

    def contract_2pt_TMD(self, prop_f, phases, trafo, tag, interpolation="5"):
        """Contract 2pt TMD correlator."""
        g.message("Begin sink smearing")
        tmp_trafo = g.convert(trafo, prop_f.grid.precision)
        prop_f = g.create.smear.boosted_smearing(tmp_trafo, prop_f, w=self.width, boost=self.pos_boost)
        g.message("Sink smearing completed")

        # Select interpolation operator
        if interpolation == "5":
            dq = g.qcd.baryon.diquark(g(prop_f * Cg5), g(Cg5 * prop_f))
        elif interpolation == "T5":
            dq = g.qcd.baryon.diquark(g(prop_f * CgT5), g(CgT5 * prop_f))
        elif interpolation == "Z5":
            dq = g.qcd.baryon.diquark(g(prop_f * CgZ5), g(CgZ5 * prop_f))
        else:
            raise ValueError("Invalid interpolation operator")

        proton1 = g(g.spin_trace(dq) * prop_f + dq * prop_f)
        prop_unit = g.mspincolor(prop_f.grid)
        prop_unit = g.identity(prop_unit)
        corr = g.slice_trDA([prop_unit], [proton1], phases, 3)
        corr = [[corr[0][i][j] for i in range(0, len(corr[0]))] for j in range(0, len(corr[0][0]))]

        if g.rank() == 0:
            save_proton_c2pt_hdf5(corr, tag, my_gammas, self.pilist)
        del corr


# ==========================
# Main script
# ==========================

# Global parameters
data_dir = "data"
lat_tag = "S8T8_gpt_local"
interpolation = "T5"
sm_tag = "1HYP_GSRC_W90_k3_" + interpolation
GEN_SIMD_WIDTH = 64
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

grid = g.grid([Ls, Ls, Ls, Lt], g.double)
U = g.convert(g.load(f"/home/jinchen/git/lat-software/PyQUDA_qTMD/test_gauge/S8T8_wilson_b6.0"), g.double)
U_prime, trafo = g.gauge_fix(U, maxiter=500, prec=1e-2)
del U_prime
trafo = g.identity(trafo)
gauge = gpt.LatticeGaugeGPT(U, GEN_SIMD_WIDTH)

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
    srcDp = Measurement.create_src_2pt(pos, trafo, U[0].grid)
    b = gpt.LatticePropagatorGPT(srcDp, GEN_SIMD_WIDTH)

    # Get forward propagator: smeared-point
    dirac.loadGauge(gauge)
    propag = core.invertPropagator(dirac, b, 1, 0)
    propag.save(f"data/propag/{lat_tag}_propag_bsm.npy")
    
    prop_exact_f = g.mspincolor(grid)
    gpt.LatticePropagatorGPT(prop_exact_f, GEN_SIMD_WIDTH, propag)

    # Contract 2pt TMD
    tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    phases_2pt = Measurement.make_mom_phases_2pt(U[0].grid, pos)
    Measurement.contract_2pt_TMD(prop_exact_f, phases_2pt, trafo, tag, interpolation)
