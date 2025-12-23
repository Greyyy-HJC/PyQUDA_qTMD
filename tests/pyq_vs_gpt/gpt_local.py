
# load python modules
import numpy as np

# load gpt modules
import gpt as g 
from tests.pyq_vs_gpt.PyQUDA_proton_qTMD_draft import proton_TMD, pyq_gamma_order #! import pyquda_gamma_ls and pyq_gamma_order for 3pt
from tests.pyq_vs_gpt.tools import *
from tests.pyq_vs_gpt.io_corr import *

# load pyquda modules
from pyquda import init
from pyquda_utils import core, gpt

# Global parameters
data_dir="tests/pyq_vs_gpt/data" # NOTE
lat_tag = "S8T8_gpt_local" # NOTE
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
grid = g.grid([Ls,Ls,Ls,Lt], g.double)
U = g.convert( g.load(f"/home/jinchen/git/lat-software/PyQUDA_qTMD/test_gauge/S8T8_wilson_b6.0"), g.double )
U_prime, trafo = g.gauge_fix(U, maxiter=500, prec=1e-2) # CG fix, to get trafo
del U_prime
trafo = g.identity(trafo)
U_hyp = U
gauge = gpt.LatticeGaugeGPT(U_hyp, GEN_SIMD_WIDTH)

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
    
    srcDp = Measurement.create_src_2pt(pos, trafo, U[0].grid)
    b = gpt.LatticePropagatorGPT(srcDp, GEN_SIMD_WIDTH)
    b.toDevice()
    # get forward propagator: smeared-point
    propag = core.invertPropagator(dirac, b, 1, 0) # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version
    prop_exact_f = g.mspincolor(grid)
    gpt.LatticePropagatorGPT(prop_exact_f, GEN_SIMD_WIDTH, propag)

    #! GPT: contract 2pt TMD
    tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    phases_2pt = Measurement.make_mom_phases_2pt(U[0].grid, pos)
    Measurement.contract_2pt_TMD(prop_exact_f, phases_2pt, trafo, tag, interpolation) # NOTE, new interpolation operator
    