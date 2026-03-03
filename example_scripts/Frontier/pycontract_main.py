# load python modules
import time
import os
import gc

import numpy as np
import cupy as cp

xp = cp

from pyquda import init, getMPIComm
from pyquda_plugins import pycontract

# Create .cache directory for QUDA tuning parameters
if not os.path.exists(".cache"):
    os.makedirs(".cache", exist_ok=True)

Ls = 64
Lt = 64

init(
    [2, 2, 2, 4],
    enable_mps=True,
    grid_map="shared",
    backend="cupy",
    resource_path=".cache",
)

from pyquda_utils import core, gamma, phase, io, source
from pyquda_utils.phase import MomentumPhase

from utils.boosted_smearing_pyquda import boosted_smearing
from utils.proton_qTMD_pyquda import proton_TMD, pyquda_gammas_order
from utils.io_corr import (
    get_sample_log_tag,
    get_c2pt_file_tag,
    get_qTMD_file_tag,
    save_qTMD_proton_hdf5_noRoll,
)
from utils.bw_seq_pycontract import create_bw_seq_pycontract
from utils.tools import srcLoc_distri_eq, mpi_print


def reorder_gamma_qgt(qgt_data):
    # mesonAllSinkTwoPoint returns gamma channels in native 0..15 order.
    # pyquda_main expects channels ordered by pyquda_gammas_order.
    return qgt_data[:, pyquda_gammas_order, :]


def release_memory(latt_info, label=""):
    # Python refs + CuPy pool blocks can hold memory across source iterations.
    gc.collect()
    if xp is cp:
        cp.cuda.runtime.deviceSynchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    if label:
        mpi_print(latt_info, f"MEM: released temporary memory ({label})")


def contract_qgt_meson_all_sink(
    latt_info,
    forward_prop,
    seq_fast,
    phases_3pt,
):
    all_sink = pycontract.mesonAllSinkTwoPoint(
        forward_prop, core.LatticePropagator(latt_info, seq_fast), gamma.Gamma(0)
    ).data
    qgt_fast = core.gatherLattice(
        xp.asnumpy(xp.einsum("qwtzyx,gwtzyx->qgt", phases_3pt, all_sink)),
        [2, -1, -1, -1],
    )
    # gatherLattice only returns data on root rank; non-root receives None.
    if qgt_fast is not None:
        qgt_fast = reorder_gamma_qgt(qgt_fast)
    return qgt_fast

# Global parameters
data_dir = "/lustre/orion/nph158/proj-shared/jinchen/debug/PyQUDA_qTMD/example_scripts/Frontier/data"  # NOTE
interpolation = "T5"  # NOTE, new interpolation operator
sm_tag = "1HYP_GSRC_W90_k3_" + interpolation  # NOTE
GEN_SIMD_WIDTH = 64
conf = 1050
lat_tag = "S64T64_pycontract_frontier"

# --------------------------
# Setup parameters
# --------------------------
parameters = {
    # NOTE:
    "eta": [0],  # irrelavant for CG TMD
    "b_z": 10,
    "b_T": 10,
    "qext": [
        [x, y, z, 0]
        for x in [-2, -1, 0, 1, 2]
        for y in [-2, -1, 0, 1, 2]
        for z in [-2, -1, 0]
    ],  # momentum transfer for TMD, pf = pi + q
    "pf": [0, 0, 9, 0],
    "p_2pt": [
        [x, y, z, 0]
        for x in [-2, -1, 0, 1, 2]
        for y in [-2, -1, 0, 1, 2]
        for z in [5, 6, 7, 8, 9]
    ],  # 2pt momentum, should match pf & pi
    "boost_in": [0, 0, 3],
    "boost_out": [0, 0, 3],
    "width": 9.0,
    "pol": ["PpUnpol"],
    "t_insert": 10,  # time separation for TMD
    "save_propagators": False,
}
pf = parameters["pf"]
pf_tag = (
    "PX"
    + str(pf[0])
    + "PY"
    + str(pf[1])
    + "PZ"
    + str(pf[2])
    + "dt"
    + str(parameters["t_insert"])
)
gammalist = ["5"]  # NOTE: temporarily only run one gamma structure
# gammalist = ["5", "T", "T5", "X", "X5", "Y", "Y5", "Z", "Z5", "I", "SXT", "SXY", "SXZ", "SYT", "SYZ", "SZT"]
Measurement = proton_TMD(parameters)

# --------------------------
# Load gauge and create inverter
# --------------------------

###################### load gauge ######################

L = [Ls, Ls, Ls, Lt]
xi_0, nu = 1.0, 1.0
mass = -0.049  # kappa = 0.12623
csw_r = 1.0372
csw_t = 1.0372
multigrid = [[4, 4, 4, 4]]

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)

dirac = core.getClover(latt_info, mass, 1e-12, 10000, xi_0, csw_r, csw_t, multigrid)
# dirac.setPrecision(sloppy=8)
# gauge = io.readNERSCGauge(
#     f"/lustre2/pion3d/jinchen/debug/PyQUDA_qTMD/example_scripts/LQ2/l6464f21b7130m00119m0322a.1050.coulomb.1e-14.HYP",
#     checksum=False,
#     link_trace=False,
#     plaquette=False,
# )  # todo: done hyp by gpt

gauge = io.readNERSCGauge("/lustre/orion/nph158/proj-shared/jinchen/ensemble/l6464f21b7130m00119m0322a.nersc.cg_high_prec/fixed_GLU/l6464f21b7130m00119m0322a.1050.coulomb.1e-14")

###################### setup source positions ######################
src_shift = np.array([0, 0, 0, 0]) + np.array([7, 11, 13, 23])
src_origin = np.array([int(conf) % L[i] for i in range(4)]) + src_shift
src_positions = srcLoc_distri_eq(L, src_origin)  # create a list of source 4*4*4*4

src_shift = np.array([0, 0, 0, 0]) + np.array([7 + 8, 11 + 8, 13 + 8, 23 + 8])
src_origin = np.array([int(conf) % L[i] for i in range(4)]) + src_shift
src_positions = src_positions + srcLoc_distri_eq(
    L, src_origin
)  # create a list of source

src_production = src_positions[
    0:2
]  # take the number of sources needed for this project NOTE


# --------------------------
# Start measurements
# --------------------------

###################### record the finished source position ######################
sample_log_file = (
    data_dir + "/sample_log_qtmd/" + str(conf) + "_" + sm_tag + "_" + pf_tag
)
if latt_info.mpi_rank == 0:
    f = open(sample_log_file, "a+")
    f.close()

#! Measurement
###################### loop over sources ######################
for ipos, pos in enumerate(src_production):

    sample_log_tag = get_sample_log_tag(str(conf), pos, sm_tag + "_" + pf_tag)
    mpi_print(latt_info, f"START: {sample_log_tag}")

    # with open(sample_log_file, "a+") as f:
    #     f.seek(0)
    #     if sample_log_tag in f.read():
    #         mpi_print(latt_info, f"SKIP: {sample_log_tag}")
            # continue  # NOTE comment this out for test otherwise it will skip all the sources that are already done

    # >>>>>>>>>>>>>>>>>>>>>>>>> Propagators <<<<<<<<<<<<<<<<<<<<<<<<<<#

    # get forward propagator boosted source
    t0 = time.time()
    srcD = source.propagator(latt_info, "point", pos)
    srcDp = boosted_smearing(srcD, w=parameters["width"], boost=parameters["boost_in"])
    
    srcDp.save(f"data/propag/{lat_tag}_srcDp.npy")

    mpi_print(latt_info, f"TIME Pyquda: Generatring boosted src {time.time() - t0}s")

    # get forward propagator: smeared-point

    t0 = time.time()
    dirac.loadGauge(gauge)
    propag = core.invertPropagator(
        dirac, srcDp, 1, 0
    )  # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version
    
    propag.save(f"data/propag/{lat_tag}_propag_bsm.npy")

    mpi_print(latt_info, f"TIME Pyquda: Forward propagator inversion {time.time() - t0}s")


    #! PyQUDA: contract 2pt TMD

    t0 = time.time()
    tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    p_2pt_xyz = [[-v[0], -v[1], -v[2]] for v in parameters["p_2pt"]]
    phases_2pt = MomentumPhase(latt_info).getPhases(p_2pt_xyz, x0=pos)

    Measurement.contract_2pt_TMD(latt_info, propag, phases_2pt, tag, interpolation)
    del phases_2pt
    release_memory(latt_info, "after 2pt contraction")

    mpi_print(latt_info, f"TIME Pyquda: Contraction 2pt (includes sink smearing) {time.time() - t0}s")

    #! PyQUDA: get backward propagator through sequential source for U and D

    t0 = time.time()
    sequential_bw_prop_down_pyq = create_bw_seq_pycontract(
        dirac,
        propag,
        pos,
        parameters["width"],
        parameters["boost_out"],
        parameters["pf"],
        parameters["t_insert"],
        parameters["pol"],
        2,
        interpolation,
    )
    sequential_bw_prop_up_pyq = create_bw_seq_pycontract(
        dirac,
        propag,
        pos,
        parameters["width"],
        parameters["boost_out"],
        parameters["pf"],
        parameters["t_insert"],
        parameters["pol"],
        1,
        interpolation,
    )

    mpi_print(latt_info, f"TIME Pyquda: Backward propagator through sequential source for U and D {time.time() - t0}s")

    #! PyQUDA: prepare phases for qext
    qext_xyz = [[v[0], v[1], v[2]] for v in parameters["qext"]]
    phases_3pt_pyq = phase.MomentumPhase(latt_info).getPhases(qext_xyz, pos)

    # >>>>>>>>>>>>>>>>>>>>>>>>> CG TMD <<<<<<<<<<<<<<<<<<<<<<<<<<#

    # prepare the TMD separate indices for CG
    W_index_list_CG_dir0, W_index_list_CG_dir1 = (
        Measurement.create_TMD_Wilsonline_index_list_CG()
    )
    W_index_list_CG = W_index_list_CG_dir0 + W_index_list_CG_dir1

    #! PyQUDA: contract TMD
    mpi_print(latt_info, "contract_TMD loop: CG no links")
    t0_contract = time.time()

    t0 = time.time()
    proton_TMDs_down = []  # [WL_indices][pol][qext][gammalist][tau]
    proton_TMDs_up = []

    g5 = xp.asarray(gamma.gamma(15))
    sequential_bw_prop_down_contracted_pyq = xp.einsum(
        "ij,pwtzyxilab,kl->pwtzyxkjba", g5, sequential_bw_prop_down_pyq.conj(), g5
    )
    sequential_bw_prop_up_contracted_pyq = xp.einsum(
        "ij,pwtzyxilab,kl->pwtzyxkjba", g5, sequential_bw_prop_up_pyq.conj(), g5
    )

    mpi_print(latt_info, f"TIME PyQUDA: contract bw prop with gamma_ls for U and D {time.time() - t0}s")

    #! PyQUDA: contract TMD +X direction
    tmd_forward_prop_dir0 = propag.copy()
    for iW, WL_indices in enumerate(W_index_list_CG_dir0):

        t0 = time.time()
        mpi_print(latt_info, f"TIME PyQUDA: contract TMD {iW+1}/{len(W_index_list_CG)} {WL_indices}")
        if iW == 0:
            WL_indices_previous = [0, 0, 0, 0]
        else:
            WL_indices_previous = W_index_list_CG_dir0[iW - 1]

        tmd_forward_prop_dir0 = Measurement.create_fw_prop_TMD_CG(
            tmd_forward_prop_dir0, WL_indices, WL_indices_previous
        )  #! note here [WL_indices] is changed to WL_indices for PyQUDA, and prop_exact_f is changed to propag
        mpi_print(latt_info, f"TIME PyQUDA: cshift {time.time() - t0}s")
        t0 = time.time()

        temp_down = []
        for seq in sequential_bw_prop_down_contracted_pyq:
            temp_down.append(
                contract_qgt_meson_all_sink(
                    latt_info,
                    tmd_forward_prop_dir0,
                    seq,
                    phases_3pt_pyq,
                )
            )
        proton_TMDs_down.append(temp_down)

        temp_up = []
        for seq in sequential_bw_prop_up_contracted_pyq:
            temp_up.append(
                contract_qgt_meson_all_sink(
                    latt_info,
                    tmd_forward_prop_dir0,
                    seq,
                    phases_3pt_pyq,
                )
            )
        proton_TMDs_up.append(temp_up)

        mpi_print(
            latt_info, f"TIME PyQUDA: contract TMD for U and D {time.time() - t0}s"
        )
    del tmd_forward_prop_dir0

    #! PyQUDA: contract TMD +Y direction
    tmd_forward_prop_dir1 = propag.copy()
    for iW, WL_indices in enumerate(W_index_list_CG_dir1):

        t0 = time.time()
        mpi_print(latt_info, f"TIME PyQUDA: contract TMD {iW+1+len(W_index_list_CG_dir0)}/{len(W_index_list_CG)} {WL_indices}")
        if iW == 0:
            WL_indices_previous = [0, 0, 0, 0]
        else:
            WL_indices_previous = W_index_list_CG_dir1[iW - 1]
        tmd_forward_prop_dir1 = Measurement.create_fw_prop_TMD_CG(
            tmd_forward_prop_dir1, WL_indices, WL_indices_previous
        )  #! note here [WL_indices] is changed to WL_indices for PyQUDA, and prop_exact_f is changed to propag

        mpi_print(latt_info, f"TIME PyQUDA: cshift {time.time() - t0}s")

        t0 = time.time()
        temp_down = []
        for seq in sequential_bw_prop_down_contracted_pyq:
            temp_down.append(
                contract_qgt_meson_all_sink(
                    latt_info,
                    tmd_forward_prop_dir1,
                    seq,
                    phases_3pt_pyq,
                )
            )
        proton_TMDs_down.append(temp_down)

        temp_up = []
        for seq in sequential_bw_prop_up_contracted_pyq:
            temp_up.append(
                contract_qgt_meson_all_sink(
                    latt_info,
                    tmd_forward_prop_dir1,
                    seq,
                    phases_3pt_pyq,
                )
            )
        proton_TMDs_up.append(temp_up)

        mpi_print(
            latt_info, f"TIME PyQUDA: contract TMD for U and D {time.time() - t0}s"
        )
    del tmd_forward_prop_dir1
    del sequential_bw_prop_down_contracted_pyq
    del sequential_bw_prop_up_contracted_pyq

    proton_TMDs_down = np.array(proton_TMDs_down)
    proton_TMDs_up = np.array(proton_TMDs_up)
    mpi_print(
        latt_info,
        f"contract_TMD over: proton_TMDs.shape {np.shape(proton_TMDs_down)} {time.time()-t0_contract}s",
    )

    # save the TMD correlators
    for i, pol in enumerate(parameters["pol"]):

        t0 = time.time()

        # reorder gamma, and cut useful tau in [src_t, src_t+tsep+2)
        if latt_info.mpi_rank == 0 and i == 0:
            proton_TMDs_down = np.roll(proton_TMDs_down, -pos[3], axis=-1)
            proton_TMDs_up = np.roll(proton_TMDs_up, -pos[3], axis=-1)
            proton_TMDs_down = proton_TMDs_down[
                :, :, :, :, : parameters["t_insert"] + 2
            ]
            proton_TMDs_up = proton_TMDs_up[:, :, :, :, : parameters["t_insert"] + 2]
        proton_TMDs_down = getMPIComm().bcast(proton_TMDs_down, root=0)
        proton_TMDs_up = getMPIComm().bcast(proton_TMDs_up, root=0)

        #! parallel the io through flavor and gamma
        tasks = []
        for gidx in range(len(gammalist)):
            tasks.append((gidx, "D"))  # Down
            tasks.append((gidx, "U"))  # Up
        rank = latt_info.mpi_rank
        n_ranks = latt_info.mpi_size
        # Each rank loops over its assigned tasks (round-robin distribution)
        for task_idx in range(rank, len(tasks), n_ranks):
            gidx, flavor = tasks[task_idx]
            gm = gammalist[gidx]
            tag = get_qTMD_file_tag(
                data_dir,
                lat_tag,
                conf,
                f"CG.{flavor}.ex",
                pos,
                f"{sm_tag}.{pf_tag}.{pol}.{gm}",
            )
            data = (
                proton_TMDs_down[:, i, :, gidx : gidx + 1, :]
                if flavor == "D"
                else proton_TMDs_up[:, i, :, gidx : gidx + 1, :]
            )
            save_qTMD_proton_hdf5_noRoll(
                data,
                tag,
                [gm],
                parameters["qext"],
                W_index_list_CG,
                parameters["t_insert"],
                latt_info,
            )

        mpi_print(latt_info, f"TIME: save TMDs for {pol} {time.time() - t0}s")
    mpi_print(latt_info, "contract_TMD DONE: CG no links")
    del proton_TMDs_down
    del proton_TMDs_up
    release_memory(latt_info, "after CG TMD")

    # >>>>>>>>>>>>>>>>>>>>>>>>> GI GPD <<<<<<<<<<<<<<<<<<<<<<<<<<#

    # prepare the TMD separate indices for GI
    W_index_list_PDF = Measurement.create_PDF_Wilsonline_index_list()

    #! PyQUDA: bw prop
    sequential_prop_down_contracted_pyq = xp.einsum(
        "ij,pwtzyxilab,kl->pwtzyxkjba", g5, sequential_bw_prop_down_pyq.conj(), g5
    )
    sequential_prop_up_contracted_pyq = xp.einsum(
        "ij,pwtzyxilab,kl->pwtzyxkjba", g5, sequential_bw_prop_up_pyq.conj(), g5
    )

    mpi_print(latt_info, "contract_PDF loop: GI with links")
    t0_contract = time.time()
    proton_PDFs_down = []  # [WL_indices][pol][qext][gammalist][tau]
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

        tmd_forward_prop_pyq = Measurement.create_fw_prop_PDF_GI(
            gauge, tmd_forward_prop_pyq, WL_indices, WL_indices_previous
        )

        #! PyQUDA: contract

        temp_down = []
        for seq in sequential_prop_down_contracted_pyq:
            temp_down.append(
                contract_qgt_meson_all_sink(
                    latt_info,
                    tmd_forward_prop_pyq,
                    seq,
                    phases_3pt_pyq,
                )
            )
        proton_PDFs_down.append(temp_down)

        temp_up = []
        for seq in sequential_prop_up_contracted_pyq:
            temp_up.append(
                contract_qgt_meson_all_sink(
                    latt_info,
                    tmd_forward_prop_pyq,
                    seq,
                    phases_3pt_pyq,
                )
            )
        proton_PDFs_up.append(temp_up)

    proton_PDFs_down = np.array(proton_PDFs_down)
    proton_PDFs_up = np.array(proton_PDFs_up)

    mpi_print(latt_info, f"contract_GI_PDF over: proton_PDFs.shape {np.shape(proton_PDFs_down)} {time.time()-t0}s") 

    # save the PDF correlators
    for i, pol in enumerate(parameters["pol"]):

        t0 = time.time()

        # reorder gamma, and cut useful tau in [src_t, src_t+tsep+2)
        if latt_info.mpi_rank == 0 and i == 0:
            proton_PDFs_down = np.roll(proton_PDFs_down, -pos[3], axis=-1)
            proton_PDFs_up = np.roll(proton_PDFs_up, -pos[3], axis=-1)
            proton_PDFs_down = proton_PDFs_down[
                :, :, :, :, : parameters["t_insert"] + 2
            ]
            proton_PDFs_up = proton_PDFs_up[:, :, :, :, : parameters["t_insert"] + 2]
        proton_PDFs_down = getMPIComm().bcast(proton_PDFs_down, root=0)
        proton_PDFs_up = getMPIComm().bcast(proton_PDFs_up, root=0)

        tasks = []
        for gidx in range(len(gammalist)):
            tasks.append((gidx, 'D'))  # Down
            tasks.append((gidx, 'U'))  # Up
        rank = latt_info.mpi_rank
        n_ranks = latt_info.mpi_size
        # Each rank loops over its assigned tasks (round-robin distribution)
        for task_idx in range(rank, len(tasks), n_ranks):
            gidx, flavor = tasks[task_idx]
            gm = gammalist[gidx]
            tag = get_qTMD_file_tag(
                data_dir,
                lat_tag,
                conf,
                f"GI_PDF.{flavor}.ex",
                pos,
                f"{sm_tag}.{pf_tag}.{pol}.{gm}",
            )
            data = (
                proton_PDFs_down[:, i, :, gidx : gidx + 1, :]
                if flavor == "D"
                else proton_PDFs_up[:, i, :, gidx : gidx + 1, :]
            )
            save_qTMD_proton_hdf5_noRoll(
                data,
                tag,
                [gm],
                parameters["qext"],
                W_index_list_PDF,
                parameters["t_insert"],
                latt_info,
            )

        mpi_print(latt_info, f"TIME: save PDFs for {pol} {time.time() - t0}s")
    mpi_print(latt_info, "contract_PDF DONE: GI with links")
    del proton_PDFs_down
    del proton_PDFs_up
    del sequential_prop_down_contracted_pyq
    del sequential_prop_up_contracted_pyq
    del tmd_forward_prop_pyq
    del sequential_bw_prop_down_pyq
    del sequential_bw_prop_up_pyq
    del phases_3pt_pyq
    del propag
    del srcDp
    del srcD
    release_memory(latt_info, "end of source")

    with open(sample_log_file, "a+") as f:
        if latt_info.mpi_rank == 0:
            f.write(sample_log_tag + "\n")
    mpi_print(latt_info, "DONE: " + sample_log_tag)
