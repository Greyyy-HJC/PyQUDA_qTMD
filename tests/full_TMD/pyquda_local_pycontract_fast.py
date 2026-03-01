# load python modules
import time
import os

import numpy as np
import cupy as cp

xp = cp

from pyquda import init, getMPIComm
from pyquda_plugins import pycontract

# Create .cache directory for QUDA tuning parameters
if not os.path.exists(".cache"):
    os.makedirs(".cache", exist_ok=True)

Ls = 8
Lt = 8

init(
    None,
    [Ls, Ls, Ls, Lt],
    enable_mps=True,
    grid_map="shared",
    backend="cupy",
    resource_path=".cache",
)

from pyquda_utils import core, gamma, phase, io, source
from pyquda_utils.phase import MomentumPhase
from pyquda_utils.source import sequential12

from utils.boosted_smearing_pyquda import boosted_smearing
from utils.bw_seq_pyquda import create_bw_seq_pyquda, create_bw_seq_pycontract
from utils.proton_qTMD_pyquda import proton_TMD, my_pyquda_gammas, pyquda_gammas_order
from utils.io_corr import (
    get_sample_log_tag,
    get_c2pt_file_tag,
    get_qTMD_file_tag,
    save_qTMD_proton_hdf5_noRoll,
)
from utils.tools import _asarray_on_queue, _get_xp_from_array, srcLoc_distri_eq, mpi_print


def create_bw_seq_pyquda_baryon_fast(
    dirac, prop, origin, sm_width, sm_boost, momentum, t_insert, pol_list, flavor, interpolator="5"
):
    if interpolator == "5":
        gamma_insert = 1j * gamma.Gamma(2) @ gamma.Gamma(8) @ gamma.Gamma(15)
    elif interpolator == "T5":
        gamma_insert = 1j * gamma.Gamma(2) @ gamma.Gamma(8) @ gamma.Gamma(7)
    elif interpolator == "Z5":
        gamma_insert = 1j * gamma.Gamma(2) @ gamma.Gamma(8) @ gamma.Gamma(11)
    else:
        raise ValueError(f"Invalid interpolator: {interpolator}")

    Pp = (gamma.Gamma(0) + gamma.Gamma(8)) / 4
    pol_proj = {"PpUnpol": Pp}
    if any(pol not in pol_proj for pol in pol_list):
        return create_bw_seq_pyquda(
            dirac, prop, origin, sm_width, sm_boost, momentum, t_insert, pol_list, flavor, interpolator
        )

    xp_local = _get_xp_from_array(prop.data)
    latt_info = prop.latt_info
    GLt = latt_info.GLt
    prop_smear = boosted_smearing(prop, w=sm_width, boost=sm_boost)

    g5 = _asarray_on_queue(gamma.gamma(15), xp_local, prop_smear.data)
    mom_phase = _asarray_on_queue(MomentumPhase(latt_info).getPhase(momentum, x0=origin), xp_local, prop_smear.data)
    t_sink = (origin[3] + t_insert) % GLt

    contract_type = pycontract.BaryonContractType.IK_JL_NM
    seq_type = (
        pycontract.BaryonSequentialType.SEQUENTIAL_I
        if flavor == 1
        else pycontract.BaryonSequentialType.SEQUENTIAL_N
    )

    dst_seq = []
    for pol in pol_list:
        src_seq = pycontract.baryonSequentialTwoPoint(
            prop_smear, prop_smear, prop_smear, contract_type, seq_type, gamma_insert, gamma_insert, pol_proj[pol]
        )
        src_seq = sequential12(src_seq, t_sink)
        seq_data = _asarray_on_queue(src_seq.data, xp_local, prop_smear.data)
        data = xp_local.einsum("ij,wtzyx,wtzyxkjba->wtzyxikab", g5, mom_phase, seq_data.conj())

        smearing_input = core.LatticePropagator(latt_info)
        smearing_input.data = data
        src = boosted_smearing(smearing_input, w=sm_width, boost=sm_boost)
        prop_smeared = core.invertPropagator(dirac, src, 1, 0)
        final_term = xp_local.einsum("wtzyxijfc,ik->wtzyxjkcf", prop_smeared.data.conj(), g5)
        dst_seq.append(final_term)

    return _asarray_on_queue(dst_seq, xp_local, prop_smear.data)


def reorder_gamma_qgt(qgt_data):
    # mesonAllSinkTwoPoint returns gamma channels in native 0..15 order.
    # pyquda_local expects channels ordered by pyquda_gammas_order.
    return qgt_data[:, pyquda_gammas_order, :]

# Global parameters
data_dir = "/home/jinchen/git/lat-software/PyQUDA_qTMD/tests/full_TMD/data"  # NOTE
interpolation = "T5"  # NOTE, new interpolation operator
sm_tag = "1HYP_GSRC_W90_k3_" + interpolation  # NOTE
GEN_SIMD_WIDTH = 64
conf = 0
lat_tag = "S8T8_pyquda_local_pycontract_fast"

# --------------------------
# Setup parameters
# --------------------------
parameters = {
    # NOTE:
    "eta": [0],  # irrelavant for CG TMD
    "b_z": 2,
    "b_T": 2,
    "qext": [
        [x, y, z, 0] for x in [2] for y in [-2] for z in [0]
    ],  # momentum transfer for TMD, pf = pi + q
    # "qext": [list(v + (0,)) for v in {tuple(sorted((x, y, z))) for x in [-2,-1,0] for y in [-2,-1,0] for z in [0]}], # momentum transfer for TMD, pf = pi + q
    "pf": [0, 0, 9, 0],
    "p_2pt": [
        [x, y, z, 0] for x in [2] for y in [-2] for z in [0]
    ],  # 2pt momentum, should match pf & pi
    "boost_in": [0, 0, 3],
    "boost_out": [0, 0, 3],
    "width": 9.0,
    "pol": ["PpUnpol"],
    "t_insert": 4,  # time separation for TMD
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
mass = -0.038888  # kappa = 0.12623
csw_r = 1.02868
csw_t = 1.02868
multigrid = None

latt_info = core.LatticeInfo([Ls, Ls, Ls, Lt], -1, xi_0 / nu)

dirac = core.getClover(latt_info, mass, 1e-12, 10000, xi_0, csw_r, csw_t, multigrid)
# dirac.setPrecision(sloppy=8)
gauge = io.readNERSCGauge(
    f"/home/jinchen/git/lat-software/PyQUDA_qTMD/test_gauge/S8T8_wilson_b6.0"
)

first_gamma = my_pyquda_gammas[0]
n_gamma = len(my_pyquda_gammas)

pyquda_gamma_ls = xp.empty(
    (n_gamma,) + first_gamma.shape,
    dtype=first_gamma.dtype,
    # device=first_gamma.device,  # key: use the same device as gamma_pyq
)

for gamma_idx, gamma_pyq in enumerate(my_pyquda_gammas):
    pyquda_gamma_ls[gamma_idx] = xp.asarray(gamma_pyq)

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
    0:1
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

    # >>>>>>>>>>>>>>>>>>>>>>>>> Propagators <<<<<<<<<<<<<<<<<<<<<<<<<<#

    # get forward propagator boosted source
    t0 = time.time()
    srcD = source.propagator(latt_info, "point", pos)
    srcDp = boosted_smearing(srcD, w=parameters["width"], boost=parameters["boost_in"])
    
    srcDp.save(f"tests/full_TMD/data/propag/{lat_tag}_srcDp.npy")

    mpi_print(latt_info, f"TIME Pyquda: Generatring boosted src {time.time() - t0}s")

    # get forward propagator: smeared-point

    t0 = time.time()
    dirac.loadGauge(gauge)
    propag = core.invertPropagator(
        dirac, srcDp, 1, 0
    )  # NOTE or "propag = core.invertPropagator(dirac, b, 0)" depends on the quda version
    
    propag.save(f"tests/full_TMD/data/propag/{lat_tag}_propag_bsm.npy")

    mpi_print(latt_info, f"TIME Pyquda: Forward propagator inversion {time.time() - t0}s")


    #! PyQUDA: contract 2pt TMD

    t0 = time.time()
    tag = get_c2pt_file_tag(data_dir, lat_tag, conf, "ex", pos, sm_tag)
    p_2pt_xyz = [[-v[0], -v[1], -v[2]] for v in parameters["p_2pt"]]
    phases_2pt = MomentumPhase(latt_info).getPhases(p_2pt_xyz, x0=pos)

    Measurement.contract_2pt_TMD(latt_info, propag, phases_2pt, tag, interpolation)

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
            temp1 = pycontract.mesonAllSinkTwoPoint(
                tmd_forward_prop_dir0, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)
            ).data
            temp2 = core.gatherLattice(
                xp.asnumpy(xp.einsum("qwtzyx,gwtzyx->qgt", phases_3pt_pyq, temp1)),
                [2, -1, -1, -1],
            )
            temp2 = reorder_gamma_qgt(temp2)
            temp_down.append(temp2)
        proton_TMDs_down.append(temp_down)

        temp_up = []
        for seq in sequential_bw_prop_up_contracted_pyq:
            temp1 = pycontract.mesonAllSinkTwoPoint(
                tmd_forward_prop_dir0, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)
            ).data
            temp2 = core.gatherLattice(
                xp.asnumpy(xp.einsum("qwtzyx,gwtzyx->qgt", phases_3pt_pyq, temp1)),
                [2, -1, -1, -1],
            )
            temp2 = reorder_gamma_qgt(temp2)
            temp_up.append(temp2)
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
            temp1 = pycontract.mesonAllSinkTwoPoint(
                tmd_forward_prop_dir1, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)
            ).data
            temp2 = core.gatherLattice(
                xp.asnumpy(xp.einsum("qwtzyx,gwtzyx->qgt", phases_3pt_pyq, temp1)),
                [2, -1, -1, -1],
            )
            temp2 = reorder_gamma_qgt(temp2)
            temp_down.append(temp2)
        proton_TMDs_down.append(temp_down)

        temp_up = []
        for seq in sequential_bw_prop_up_contracted_pyq:
            temp1 = pycontract.mesonAllSinkTwoPoint(
                tmd_forward_prop_dir1, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)
            ).data
            temp2 = core.gatherLattice(
                xp.asnumpy(xp.einsum("qwtzyx,gwtzyx->qgt", phases_3pt_pyq, temp1)),
                [2, -1, -1, -1],
            )
            temp2 = reorder_gamma_qgt(temp2)
            temp_up.append(temp2)
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
            temp1 = pycontract.mesonAllSinkTwoPoint(
                tmd_forward_prop_pyq, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)
            ).data
            temp2 = core.gatherLattice(
                xp.asnumpy(xp.einsum("qwtzyx,gwtzyx->qgt", phases_3pt_pyq, temp1)),
                [2, -1, -1, -1],
            )
            temp2 = reorder_gamma_qgt(temp2)
            temp_down.append(temp2)
        proton_PDFs_down.append(temp_down)

        temp_up = []
        for seq in sequential_prop_up_contracted_pyq:
            temp1 = pycontract.mesonAllSinkTwoPoint(
                tmd_forward_prop_pyq, core.LatticePropagator(latt_info, seq), gamma.Gamma(0)
            ).data
            temp2 = core.gatherLattice(
                xp.asnumpy(xp.einsum("qwtzyx,gwtzyx->qgt", phases_3pt_pyq, temp1)),
                [2, -1, -1, -1],
            )
            temp2 = reorder_gamma_qgt(temp2)
            temp_up.append(temp2)
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

    with open(sample_log_file, "a+") as f:
        if latt_info.mpi_rank == 0:
            f.write(sample_log_tag + "\n")
    mpi_print(latt_info, "DONE: " + sample_log_tag)
