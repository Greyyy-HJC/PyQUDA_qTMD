'''
Modified by Jinchen He, 2025-11-21.
Fixed SYCL Queue Mismatch by ensuring all arrays share the propagator's queue.
Refactored by Gemini to use pyquda_utils.source.sequential12 for time slicing.
'''
import numpy as np

from pyquda.field import LatticePropagator
from pyquda_plugins import pycontract
from pyquda_utils import core, gamma
from pyquda_utils.phase import MomentumPhase

from pyquda_utils.source import sequential12 
from utils.boosted_smearing_pyquda import boosted_smearing
from utils.tools import _get_xp_from_array, _asarray_on_queue

# ---------- Precompute Constant Spin Matrices ----------
Cg5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(15)
CgT5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(7)
CgZ5 = (1j * gamma.gamma(2) @ gamma.gamma(8)) @ gamma.gamma(11)

Pp = (gamma.gamma(0) + gamma.gamma(8)) * 0.25
Szp = (gamma.gamma(0) - 1j*gamma.gamma(1) @ gamma.gamma(2))
Szm = (gamma.gamma(0) + 1j*gamma.gamma(1) @ gamma.gamma(2))
Sxp = (gamma.gamma(0) - 1j*gamma.gamma(2) @ gamma.gamma(4))
Sxm = (gamma.gamma(0) + 1j*gamma.gamma(2) @ gamma.gamma(4))
PpSzp = Pp @ Szp
PpSzm = Pp @ Szm
PpSxp = Pp @ Sxp
PpSxm = Pp @ Sxm

PolProjections = {
    "PpSzp": PpSzp,
    "PpSzm": PpSzm,
    "PpSxp": PpSxp,
    "PpSxm": PpSxm,  
    "PpUnpol": Pp,  
}



_INTERPOLATOR_GAMMA = {
    "5": 1j * gamma.Gamma(2) @ gamma.Gamma(8) @ gamma.Gamma(15),
    "T5": 1j * gamma.Gamma(2) @ gamma.Gamma(8) @ gamma.Gamma(7),
    "Z5": 1j * gamma.Gamma(2) @ gamma.Gamma(8) @ gamma.Gamma(11),
}

_I = pycontract.BaryonSequentialType.SEQUENTIAL_I
_J = pycontract.BaryonSequentialType.SEQUENTIAL_J
_N = pycontract.BaryonSequentialType.SEQUENTIAL_N
_C = pycontract.BaryonContractType
_BW_SEQ_COMB = {
    # flavor=2: down insertion
    2: [
        (0.25, _I, _C.IK_JL_NM),
        (0.25, _I, _C.IK_JM_NL),
        (-0.25, _I, _C.IL_JK_NM),
        (-0.25, _I, _C.IL_JM_NK),
        (0.25, _J, _C.IK_JL_NM),
        (-0.25, _J, _C.IL_JK_NM),
        (-0.25, _J, _C.IM_JK_NL),
        (0.25, _J, _C.IM_JL_NK),
    ],
    # flavor=1: up insertion
    1: [
        (0.25, _I, _C.IK_JL_NM),
        (-0.25, _I, _C.IL_JK_NM),
        (-0.25, _I, _C.IM_JK_NL),
        (0.25, _I, _C.IM_JL_NK),
        (0.25, _J, _C.IK_JL_NM),
        (-0.25, _J, _C.IL_JK_NM),
        (0.25, _J, _C.IK_JM_NL),
        (-0.25, _J, _C.IL_JM_NK),
        (0.5, _N, _C.IK_JL_NM),
        (-0.5, _N, _C.IL_JK_NM),
        (0.25, _N, _C.IK_JM_NL),
        (-0.25, _N, _C.IL_JM_NK),
        (-0.25, _N, _C.IM_JK_NL),
        (0.25, _N, _C.IM_JL_NK),
    ],
}

def _g(index, factor=1):
    return gamma.Gamma(index, factor)


# Merge polarization terms into pycontract-compatible components.
# Each component is Gamma or Polarize (sum of two Gamma terms).
_POLARIZATION_COMPONENTS = {
    "PpUnpol": [_g(0, 0.25) + _g(8, 0.25)],
    "PpSzp": [_g(0, 0.25) + _g(3, -0.25j), _g(8, 0.25) + _g(11, -0.25j)],
    "PpSzm": [_g(0, 0.25) + _g(3, 0.25j), _g(8, 0.25) + _g(11, 0.25j)],
    "PpSxp": [_g(0, 0.25) + _g(6, -0.25j), _g(8, 0.25) + _g(14, -0.25j)],
    "PpSxm": [_g(0, 0.25) + _g(6, 0.25j), _g(8, 0.25) + _g(14, 0.25j)],
}


def _build_seq_source_pycontract(prop_smear, gamma_insert, polarization_components, flavor):
    latt_info = prop_smear.latt_info
    xp_local = _get_xp_from_array(prop_smear.data)
    seq_sum = None
    for polarization_component in polarization_components:
        for wick_coef, seq_type, contract_type in _BW_SEQ_COMB[flavor]:
            seq_raw = pycontract.baryonSequentialTwoPoint(
                prop_smear,
                prop_smear,
                prop_smear,
                contract_type,
                seq_type,
                gamma_insert,
                gamma_insert,
                polarization_component,
            )
            term = wick_coef * seq_raw.data
            seq_sum = term if seq_sum is None else (seq_sum + term)
    # Match the spin/color convention of the reference implementation.
    seq_sum = -xp_local.swapaxes(xp_local.swapaxes(seq_sum, -4, -3), -2, -1)
    seq_prop = core.LatticePropagator(latt_info)
    seq_prop.data = seq_sum
    return seq_prop


def create_bw_seq_pycontract(
    dirac, prop, origin, sm_width, sm_boost, momentum, t_insert, pol_list, flavor, interpolator="5"
):
    if interpolator not in _INTERPOLATOR_GAMMA:
        raise ValueError(f"Invalid interpolator: {interpolator}")
    if flavor not in _BW_SEQ_COMB:
        raise ValueError(f"Invalid flavor: {flavor}")
    unsupported_pols = [pol for pol in pol_list if pol not in _POLARIZATION_COMPONENTS]
    if unsupported_pols:
        raise ValueError(
            "Unsupported polarization(s) for pycontract fast path: "
            f"{unsupported_pols}. Supported: {sorted(_POLARIZATION_COMPONENTS.keys())}"
        )

    gamma_insert = _INTERPOLATOR_GAMMA[interpolator]
    xp_local = _get_xp_from_array(prop.data)
    latt_info = prop.latt_info
    GLt = latt_info.GLt
    prop_smear = boosted_smearing(prop, w=sm_width, boost=sm_boost)

    g5 = _asarray_on_queue(gamma.gamma(15), xp_local, prop_smear.data)
    mom_phase = _asarray_on_queue(
        MomentumPhase(latt_info).getPhase(momentum, x0=origin), xp_local, prop_smear.data
    )
    t_sink = (origin[3] + t_insert) % GLt

    dst_seq = []
    for pol in pol_list:
        seq_source = _build_seq_source_pycontract(
            prop_smear, gamma_insert, _POLARIZATION_COMPONENTS[pol], flavor
        )
        seq_data = _asarray_on_queue(sequential12(seq_source, t_sink).data, xp_local, prop_smear.data)
        data = xp_local.einsum("ij,wtzyx,wtzyxkjba->wtzyxikab", g5, mom_phase, seq_data.conj())

        smearing_input = core.LatticePropagator(latt_info)
        smearing_input.data = data
        src = boosted_smearing(smearing_input, w=sm_width, boost=sm_boost)
        prop_smeared = core.invertPropagator(dirac, src, 1, 0)
        final_term = xp_local.einsum("wtzyxijfc,ik->wtzyxjkcf", prop_smeared.data.conj(), g5)
        dst_seq.append(final_term)

    return _asarray_on_queue(dst_seq, xp_local, prop_smear.data)



