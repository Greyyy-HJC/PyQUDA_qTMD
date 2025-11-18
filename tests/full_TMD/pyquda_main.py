
# load python modules
import time

import numpy as np
import cupy as cp

from types import SimpleNamespace
from opt_einsum import contract

from pyquda import init, getMPIComm
from pyquda_utils import core, gamma, phase, io, source
from pyquda_utils.phase import MomentumPhase
from pyquda_plugins import pycontract #: for PyQUDA contraction v2

from utils.boosted_smearing_pyquda import boosted_smearing
from utils.bw_seq_pyquda import create_bw_seq_pyquda
from utils.proton_qTMD_pyquda import proton_TMD, pyquda_gammas_order
from utils.io_corr import get_sample_log_tag, get_c2pt_file_tag, get_qTMD_file_tag, save_qTMD_proton_hdf5_noRoll
from utils.tools import srcLoc_distri_eq, mpi_print, _get_xp_from_array, _ensure_backend


# Global parameters
data_dir="/home/jinchen/git/lat-software/PyQUDA_qTMD/tests/full_TMD/data" # NOTE
lat_tag = "l64c64a076" # NOTE
interpolation = "T5" # NOTE, new interpolation operator
sm_tag = "1HYP_GSRC_W90_k3_"+interpolation # NOTE
GEN_SIMD_WIDTH = 64

# --------------------------
# initiate quda
# --------------------------
mpi_geometry = [1, 1, 1, 1]
init(mpi_geometry, enable_mps=True, grid_map="shared")
G5 = gamma.gamma(15)

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
    "qext_PDF": [[x,y,z,0] for x in [2] for y in [-2] for z in [0]], # momentum transfer for PDF, not used 
    "pf": [0,0,0,0],
    "p_2pt": [[x,y,z,0] for x in [2] for y in [-2] for z in [0]], # 2pt momentum, should match pf & pi

    "boost_in": [0,0,0],
    "boost_out": [0,0,0],
    "width" : 1.0,

    "pol": ["PpUnpol"],
    "t_insert": 0, # time separation for TMD

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
Ls = 8
Lt = 8
L = [Ls, Ls, Ls, Lt]
xi_0, nu = 1.0, 1.0
mass = -0.038888 # kappa = 0.12623
csw_r = 1.02868
csw_t = 1.02868
multigrid = None 

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)

if latt_info.mpi_rank == 0:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_num", type=int, default=0, help="Configuration number")
    args, unknown = parser.parse_known_args()
    conf = args.config_num

    print(f"--lat_tag {lat_tag}")
    print(f"--sm_tag {sm_tag}")
    print(f"--config_num {conf}")


dirac = core.getClover(latt_info, mass, 1e-8, 10000, xi_0, csw_r, csw_t, multigrid)
gauge = io.readNERSCGauge(f"/home/jinchen/git/lat-software/PyQUDA_qTMD/test_gauge/S8T8_wilson_b6.0")

identity_data = cp.zeros((Lt,Ls,Ls,Ls,3,3), dtype=cp.complex128)
identity_data[..., 0,0] = 1
identity_data[..., 1,1] = 1
identity_data[..., 2,2] = 1
identity_trafo = SimpleNamespace(data=identity_data, latt_info=latt_info)

# gauge.hypSmear(1, 0.75, 0.6, 0.3, -1)

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

###################### create multigrid inverter ######################

if latt_info.mpi_rank == 0:
    print("DEBUG plaquette U_hyp:", gauge.plaquette())


# --------------------------
# Start measurements
# --------------------------

###################### record the finished source position ######################
sample_log_file = data_dir + "/sample_log_qtmd/" + str(conf) + '_' + sm_tag + "_" + pf_tag
if latt_info.mpi_rank == 0:
    f = open(sample_log_file, "a+")
    f.close()

#! Measurement
###################### loop over sources ######################
for ipos, pos in enumerate(src_production):
    
    sample_log_tag = get_sample_log_tag(str(conf), pos, sm_tag + "_" + pf_tag)
    if latt_info.mpi_rank == 0:
        print(f"START: {sample_log_tag}")
    with open(sample_log_file, "a+") as f:
        f.seek(0)
        if sample_log_tag in f.read():
            if latt_info.mpi_rank == 0:
                print("SKIP: " + sample_log_tag)
            continue # NOTE comment this out for test otherwise it will skip all the sources that are already done



    #>>>>>>>>>>>>>>>>>>>>>>>>> Propagators <<<<<<<<<<<<<<<<<<<<<<<<<<#

    # get forward propagator boosted source
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    srcD = source.propagator(latt_info, "point", pos)
    srcDp = boosted_smearing(identity_trafo, srcD, w=parameters["width"], boost=parameters["boost_in"])
    cp.cuda.runtime.deviceSynchronize()
    if latt_info.mpi_rank == 0:
        print("TIME Pyquda: Generatring boosted src", time.time() - t0)

    # get forward propagator: smeared-point
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    dirac.loadGauge(gauge) #TODO: debug
    propag = core.invertPropagator(dirac, srcDp, 1, 0) # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version
    cp.cuda.runtime.deviceSynchronize()
    if latt_info.mpi_rank == 0:
        print("TIME Pyquda: Forward propagator inversion", time.time() - t0)

    #! GPT: contract 2pt TMD
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    phases_2pt = MomentumPhase(latt_info).getPhases([pos], x0=pos)
    
    # Measurement.contract_2pt_TMD(latt_info, propag, phases_2pt, identity_trafo, tag, interpolation)

    cp.cuda.runtime.deviceSynchronize()
    if latt_info.mpi_rank == 0:
        print("TIME Pyquda: Contraction 2pt (includes sink smearing)", time.time() - t0)
        
    
    #! PyQUDA: get backward propagator through sequential source for U and D
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    sequential_bw_prop_down_pyq = create_bw_seq_pyquda(dirac, propag, identity_trafo, pos, parameters["width"], parameters["boost_out"], parameters["pf"], parameters["t_insert"], parameters["pol"], 2, interpolation)
    sequential_bw_prop_up_pyq = create_bw_seq_pyquda(dirac, propag, identity_trafo, pos, parameters["width"], parameters["boost_out"], parameters["pf"], parameters["t_insert"], parameters["pol"], 1, interpolation)
    
    cp.cuda.runtime.deviceSynchronize()
    if latt_info.mpi_rank == 0:
        print("TIME Pyquda: Backward propagator through sequential source for U and D", time.time() - t0)

    #! PyQUDA: prepare phases for qext
    qext_xyz = [[v[0], v[1], v[2]] for v in parameters["qext"]]
    phases_3pt_pyq = phase.MomentumPhase(latt_info).getPhases(qext_xyz, pos)
    
    qext_pdf_xyz = [[v[0], v[1], v[2]] for v in parameters["qext_PDF"]]
    phase_PDF = MomentumPhase(latt_info).getPhases(qext_pdf_xyz, pos)
    
    
    

    #>>>>>>>>>>>>>>>>>>>>>>>>> CG TMD <<<<<<<<<<<<<<<<<<<<<<<<<<#

    # prepare the TMD separate indices for CG
    W_index_list_CG_dir0, W_index_list_CG_dir1 = Measurement.create_TMD_Wilsonline_index_list_CG()
    W_index_list_CG = W_index_list_CG_dir0 + W_index_list_CG_dir1
    
    #! PyQUDA: contract TMD
    if latt_info.mpi_rank == 0:
        print("contract_TMD loop: CG no links")
    t0_contract = time.time()
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.time()
    proton_TMDs_down = [] # [WL_indices][pol][qext][gammalist][tau]
    proton_TMDs_up = []
    
    sequential_bw_prop_down_contracted_pyq = contract(
                "ij, pwtzyxilab, kl -> pwtzyxkjba",
                G5, sequential_bw_prop_down_pyq.conj(), G5
            )

    sequential_bw_prop_up_contracted_pyq = contract(
                "ij, pwtzyxilab, kl -> pwtzyxkjba",
                G5, sequential_bw_prop_up_pyq.conj(), G5
            )
    
    cp.cuda.runtime.deviceSynchronize()
    if latt_info.mpi_rank == 0:
        print(f"TIME PyQUDA: contract bw prop with gamma_ls for U and D", time.time() - t0)
   
    #! PyQUDA: contract TMD +X direction
    tmd_forward_prop_dir0 = propag.copy()
    for iW, WL_indices in enumerate(W_index_list_CG_dir0):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        if latt_info.mpi_rank == 0:
            print(f"TIME PyQUDA: contract TMD {iW+1}/{len(W_index_list_CG)} {WL_indices}")
        if iW == 0:
            WL_indices_previous = [0, 0, 0, 0]
        else:
            WL_indices_previous = W_index_list_CG_dir0[iW - 1]
            
        tmd_forward_prop_dir0 = Measurement.create_fw_prop_TMD_CG(tmd_forward_prop_dir0, WL_indices, WL_indices_previous) #! note here [WL_indices] is changed to WL_indices for PyQUDA, and prop_exact_f is changed to propag
        cp.cuda.runtime.deviceSynchronize()
        if latt_info.mpi_rank == 0:
            print(f"TIME PyQUDA: cshift", time.time() - t0)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        temp_down = []
        for seq in sequential_bw_prop_down_contracted_pyq:
            temp1 = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop_dir0, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)).data # loop over 16 gamma structure
            temp2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> qgt", phases_3pt_pyq, temp1).get(), [2, -1, -1, -1])
            temp_down.append(temp2)
        proton_TMDs_down.append(temp_down)
        
        temp_up = []
        for seq in sequential_bw_prop_up_contracted_pyq:
            temp1 = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop_dir0, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)).data # loop over 16 gamma structure
            temp2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> qgt", phases_3pt_pyq, temp1).get(), [2, -1, -1, -1])
            temp_up.append(temp2)
        proton_TMDs_up.append(temp_up)
        cp.cuda.runtime.deviceSynchronize()
        if latt_info.mpi_rank == 0:
            print(f"TIME PyQUDA: contract TMD for U and D", time.time() - t0)
    del tmd_forward_prop_dir0
        
    #! PyQUDA: contract TMD +Y direction
    tmd_forward_prop_dir1 = propag.copy()
    for iW, WL_indices in enumerate(W_index_list_CG_dir1):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        if latt_info.mpi_rank == 0:
            print(f"TIME PyQUDA: contract TMD {iW+1+len(W_index_list_CG_dir0)}/{len(W_index_list_CG)} {WL_indices}")
        if iW == 0:
            WL_indices_previous = [0, 0, 0, 0]
        else:
            WL_indices_previous = W_index_list_CG_dir1[iW - 1]
        tmd_forward_prop_dir1 = Measurement.create_fw_prop_TMD_CG(tmd_forward_prop_dir1, WL_indices, WL_indices_previous) #! note here [WL_indices] is changed to WL_indices for PyQUDA, and prop_exact_f is changed to propag
        cp.cuda.runtime.deviceSynchronize()
        if latt_info.mpi_rank == 0:
            print(f"TIME PyQUDA: cshift", time.time() - t0)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        temp_down = []
        for seq in sequential_bw_prop_down_contracted_pyq:
            temp1 = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop_dir1, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)).data
            temp2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> qgt", phases_3pt_pyq, temp1).get(), [2, -1, -1, -1])
            temp_down.append(temp2)
        proton_TMDs_down.append(temp_down)
        
        temp_up = []
        for seq in sequential_bw_prop_up_contracted_pyq:
            temp1 = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop_dir1, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)).data
            temp2 = core.gatherLattice(contract("qwtzyx, gwtzyx -> qgt", phases_3pt_pyq, temp1).get(), [2, -1, -1, -1])
            temp_up.append(temp2)
        proton_TMDs_up.append(temp_up)
        cp.cuda.runtime.deviceSynchronize()
        if latt_info.mpi_rank == 0:
            print(f"TIME PyQUDA: contract TMD for U and D", time.time() - t0)
    del tmd_forward_prop_dir1
    del sequential_bw_prop_down_contracted_pyq
    del sequential_bw_prop_up_contracted_pyq
    
    proton_TMDs_down = np.array(proton_TMDs_down)
    proton_TMDs_up = np.array(proton_TMDs_up)
    if latt_info.mpi_rank == 0:
        print(f"contract_TMD over: proton_TMDs.shape {np.shape(proton_TMDs_down)} {time.time()-t0_contract}s")

    # save the TMD correlators
    for i, pol in enumerate(parameters["pol"]):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()

        # reorder gamma, and cut useful tau in [src_t, src_t+tsep+2)
        if latt_info.mpi_rank == 0 and i == 0:
            proton_TMDs_down = np.roll(proton_TMDs_down, -pos[3], axis=-1)
            proton_TMDs_up = np.roll(proton_TMDs_up, -pos[3], axis=-1)
            proton_TMDs_down = proton_TMDs_down[:,:,:,pyquda_gammas_order,:parameters["t_insert"]+2]
            proton_TMDs_up = proton_TMDs_up[:,:,:,pyquda_gammas_order,:parameters["t_insert"]+2]
        proton_TMDs_down = getMPIComm().bcast(proton_TMDs_down, root=0)
        proton_TMDs_up = getMPIComm().bcast(proton_TMDs_up, root=0)

        #! parallel the io through flavor and gamma
        tasks = []
        for gidx in range(len(gammalist)):
            tasks.append((gidx, 'D'))  # Down
            tasks.append((gidx, 'U'))  # Up
        rank = latt_info.mpi_rank
        if rank < len(tasks):
            gidx, flavor = tasks[rank]
            gm = gammalist[gidx]
            tag = get_qTMD_file_tag(data_dir, lat_tag, conf, f"CG.{flavor}.ex", pos, f"{sm_tag}.{pf_tag}.{pol}.{gm}")
            print(f"DEBUG: rank {rank}, {tag}")
            data = proton_TMDs_down[:, i, :, gidx:gidx+1, :] if flavor == 'D' else proton_TMDs_up[:, i, :, gidx:gidx+1, :]
            save_qTMD_proton_hdf5_noRoll(data, tag, [gm], parameters["qext"], W_index_list_CG, parameters["t_insert"], latt_info)
        cp.cuda.runtime.deviceSynchronize()
        if latt_info.mpi_rank == 0:
            print(f"TIME: save TMDs for {pol}", time.time() - t0)
    if latt_info.mpi_rank == 0:
        print("contract_TMD DONE: CG no links")
    

    #>>>>>>>>>>>>>>>>>>>>>>>>> GI GPD <<<<<<<<<<<<<<<<<<<<<<<<<<#

    # prepare the TMD separate indices for GI
    W_index_list_PDF = Measurement.create_PDF_Wilsonline_index_list()
    
    #! PyQUDA: prepare phases for qext
    qext_pdf_xyz = [[v[0], v[1], v[2]] for v in parameters["qext_PDF"]]
    phases_pdf_pyq = phase.MomentumPhase(latt_info).getPhases(qext_pdf_xyz, pos)
    
    #! PyQUDA: bw prop
    sequential_prop_down = contract(
        "ij, pwtzyxilab, kl -> pwtzyxkjba",
        G5,
        sequential_bw_prop_down_pyq.conj(),
        G5,
    )
    sequential_prop_up = contract(
        "ij, pwtzyxilab, kl -> pwtzyxkjba",
        G5,
        sequential_bw_prop_up_pyq.conj(),
        G5,
    )
    
    if latt_info.mpi_rank == 0:
        print("contract_PDF loop: GI with links")
    t0_contract = time.time()
    proton_PDFs_down = [] # [WL_indices][pol][qext][gammalist][tau]
    proton_PDFs_up = []
    for iW, WL_indices in enumerate(W_index_list_PDF):

        t0 = time.time()

        if WL_indices[1] == 0:
            WL_indices_previous = [0, 0, 0, 0]
            tmd_forward_prop_pyq = propag.copy()
        elif WL_indices[1] > 0:
            WL_indices_previous = W_index_list_PDF[iW - 1]
        elif WL_indices[1] == -1:
            WL_indices_previous = [0, 0, 0, 0]
            tmd_forward_prop_pyq = propag.copy()
        elif WL_indices[1] < -1:
            WL_indices_previous = W_index_list_PDF[iW - 1]

        tmd_forward_prop_pyq = Measurement.create_fw_prop_PDF_GI(gauge, tmd_forward_prop_pyq, WL_indices, WL_indices_previous)

        #! PyQUDA: contract
        temp_down = []
        for seq in sequential_prop_down:
            seq_lp = core.LatticePropagator(latt_info, seq)
            temp1 = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop_pyq, seq_lp, gamma.Gamma(0)).data
            contracted = contract("qwtzyx, gwtzyx -> qgt", phases_pdf_pyq, temp1).get()
            gathered = core.gatherLattice(contracted, [2, -1, -1, -1])
            temp_down.append(gathered)
        proton_PDFs_down.append(temp_down)

        temp_up = []
        for seq in sequential_prop_up:
            seq_lp = core.LatticePropagator(latt_info, seq)
            temp1 = pycontract.mesonAllSinkTwoPoint(tmd_forward_prop_pyq, seq_lp, gamma.Gamma(0)).data
            contracted = contract("qwtzyx, gwtzyx -> qgt", phases_pdf_pyq, temp1).get()
            gathered = core.gatherLattice(contracted, [2, -1, -1, -1])
            temp_up.append(gathered)
        proton_PDFs_up.append(temp_up)
        if latt_info.mpi_rank == 0:
            print(f"contract_GI_PDF over: proton_PDFs.shape {np.shape(proton_PDFs_down)} {time.time()-t0}s")

    if latt_info.mpi_rank == 0:
        print(f"TIME PyQUDA: contract GI PDF for U and D", time.time() - t0_contract)

    # save the PDF correlators
    for i, pol in enumerate(parameters["pol"]):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.time()

        # reorder gamma, and cut useful tau in [src_t, src_t+tsep+2)
        if latt_info.mpi_rank == 0 and i == 0:
            proton_PDFs_down = np.roll(proton_PDFs_down, -pos[3], axis=-1)
            proton_PDFs_up = np.roll(proton_PDFs_up, -pos[3], axis=-1)
            proton_PDFs_down = proton_PDFs_down[:,:,:,pyquda_gammas_order,:parameters["t_insert"]+2]
            proton_PDFs_up = proton_PDFs_up[:,:,:,pyquda_gammas_order,:parameters["t_insert"]+2]
        proton_PDFs_down = getMPIComm().bcast(proton_PDFs_down, root=0)
        proton_PDFs_up = getMPIComm().bcast(proton_PDFs_up, root=0)

        tasks = ['D', 'U']
        if latt_info.mpi_rank < len(tasks):
            flavor = tasks[latt_info.mpi_rank]
            tag = get_qTMD_file_tag(data_dir, lat_tag, conf, f"GI_PDF.{flavor}.ex", pos, f"{sm_tag}.{pf_tag}.{pol}")
            data = proton_PDFs_down[:, i, :, :, :] if flavor == 'D' else proton_PDFs_up[:, i, :, :, :]
            save_qTMD_proton_hdf5_noRoll(data, tag, gammalist, parameters["qext_PDF"], W_index_list_PDF, parameters["t_insert"], latt_info)

        cp.cuda.runtime.deviceSynchronize()
        if latt_info.mpi_rank == 0:
            print(f"TIME: save PDFs for {pol}", time.time() - t0)
    if latt_info.mpi_rank == 0:
        print("contract_PDF DONE: GI with links")

    with open(sample_log_file, "a+") as f:
        if latt_info.mpi_rank == 0:
            f.write(sample_log_tag+"\n")
    if latt_info.mpi_rank == 0:
        print("DONE: " + sample_log_tag)

