
# load python modules
import numpy as np

from pyquda import init
from pyquda_utils import core, gamma, io, source
from pyquda_utils.phase import MomentumPhase

from utils.boosted_smearing_pyquda import boosted_smearing
from utils.proton_qTMD_pyquda import proton_TMD
from utils.io_corr import get_sample_log_tag, get_c2pt_file_tag
from utils.tools import srcLoc_distri_eq


# Global parameters
data_dir="tests/pyq_vs_gpt/data" # NOTE
lat_tag = "S8T8_pyquda_local" # NOTE
interpolation = "T5" # NOTE, new interpolation operator
sm_tag = "1HYP_GSRC_W90_k3_"+interpolation # NOTE
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
    
    # NOTE:
    "eta": [0],  # irrelavant for CG TMD
    "b_z": 2,
    "b_T": 2,

    "qext": [[x,y,z,0] for x in [2] for y in [-2] for z in [0]], # momentum transfer for TMD, pf = pi + q
    #"qext": [list(v + (0,)) for v in {tuple(sorted((x, y, z))) for x in [-2,-1,0] for y in [-2,-1,0] for z in [0]}], # momentum transfer for TMD, pf = pi + q
    "pf": [0,0,9,0],
    "p_2pt": [[x,y,z,0] for x in [2] for y in [-2] for z in [0]], # 2pt momentum, should match pf & pi

    "boost_in": [0,0,3],
    "boost_out": [0,0,3],
    "width" : 9.0,

    "pol": ["PpUnpol"],
    "t_insert": 4, # time separation for TMD

    "save_propagators": False,
}
pf = parameters["pf"]
pf_tag = "PX"+str(pf[0]) + "PY"+str(pf[1]) + "PZ"+str(pf[2]) + "dt" + str(parameters["t_insert"])
gammalist = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
Measurement = proton_TMD(parameters)


# --------------------------
# Load gauge and create inverter
# --------------------------

###################### load gauge ######################

L = [Ls, Ls, Ls, Lt]
xi_0, nu = 1.0, 1.0
mass = -0.038888 # kappa = 0.12623
csw_r = 1.02868
csw_t = 1.02868
multigrid = None 

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-10, 10000, xi_0, csw_r, csw_t, multigrid)
dirac.setPrecision(sloppy=8)
gauge = io.readNERSCGauge(f"/home/jinchen/git/lat-software/PyQUDA_qTMD/test_gauge/S8T8_wilson_b6.0")

###################### setup source positions ######################
src_shift = np.array([0,0,0,0]) + np.array([7,11,13,23])
src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin) # create a list of source 4*4*4*4

src_shift = np.array([0,0,0,0]) + np.array([7+8,11+8,13+8,23+8])
src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
src_positions = src_positions + srcLoc_distri_eq(L, src_origin) # create a list of source

src_shift = np.array([0,0,0,0]) + np.array([7+8,11+8,13+8,23+4])
src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
src_positions = src_positions + srcLoc_distri_eq(L, src_origin) # create a list of source

src_shift = np.array([0,0,0,0]) + np.array([7+8,11+8,13+4,23+4])
src_origin = np.array([int(conf)%L[i] for i in range(4)]) + src_shift
src_positions = src_positions + srcLoc_distri_eq(L, src_origin) # create a list of source

src_production = src_positions[0:1] # take the number of sources needed for this project NOTE

# --------------------------
# Start measurements
# --------------------------


#! Measurement
###################### loop over sources ######################
for ipos, pos in enumerate(src_production):

    srcD = source.propagator(latt_info, "point", pos)
    srcDp = boosted_smearing(srcD, w=parameters["width"], boost=parameters["boost_in"])
    dirac.loadGauge(gauge) #TODO: debug
    propag = core.invertPropagator(dirac, srcDp, 1, 0) # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version
    
    
    tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    p_2pt_xyz = [[-v[0], -v[1], -v[2]] for v in parameters["p_2pt"]]
    phases_2pt = MomentumPhase(latt_info).getPhases(p_2pt_xyz, x0=pos)
    
    Measurement.contract_2pt_TMD(latt_info, propag, phases_2pt, tag, interpolation)