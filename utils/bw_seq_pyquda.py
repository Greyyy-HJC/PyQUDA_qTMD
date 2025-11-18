'''
Modified by Jinchen He, 2025-11-13.

This is the module to implement the backward sequential source in PyQUDA.
'''
import numpy as np

from pyquda.field import LatticePropagator
from pyquda_utils import core, gamma
from pyquda_utils.phase import MomentumPhase
from pyquda.field import evenodd
from utils.boosted_smearing_pyquda import boosted_smearing
from utils.tools import _get_xp_from_array, _ensure_backend

# ---------- Backend Helpers (consistent with boosted_smearing_pyquda) ----------
# def _get_xp_from_array(a):
#     """Return the base module of the array's type, e.g. cupy / numpy / dpnp / torch."""
#     # Handle case where data might be wrapped or None, default to numpy if unsure
#     if a is None:
#         return __import__("numpy")
#     base = type(a).__module__.split('.')[0]
#     return __import__(base)

# def _ensure_backend(x, xp):
#     """Move x to the same backend as xp if needed."""
#     # Check if x is already on the correct backend
#     if type(x).__module__.split('.')[0] == xp.__name__:
#         return x
#     # Handle transfer
#     if hasattr(xp, "asarray"):
#         return xp.asarray(x)
#     if xp.__name__ == "torch":
#         return xp.as_tensor(x)
#     return xp.array(x)

# ---------- Precompute Constant Spin Matrices (Lazy loading recommended, but keeping global for now) ----------
# Note: These are standard numpy arrays initially. They will be converted to 'xp' inside functions.
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


def create_bw_seq_pyquda(dirac, prop: LatticePropagator, trafo, origin, sm_width, sm_boost, momentum, t_insert, pol_list, flavor, interpolator="5"):
    """
    PyQUDA version: Build backward sequential source (Backend Agnostic).
    """
    
    if interpolator == "5":
        gamma_insert = Cg5
    elif interpolator == "T5":
        gamma_insert = CgT5
    elif interpolator == "Z5":
        gamma_insert = CgZ5
    else:
        raise ValueError(f"Invalid interpolator: {interpolator}")
    
    # 1. Identify Backend from input prop
    xp = _get_xp_from_array(prop.data)

    latt_info = prop.latt_info
    Lt = latt_info.Lt
    
    # Perform boosted smearing (boosted_smearing handles backend internally)
    prop = boosted_smearing(trafo, prop, w=sm_width, boost=sm_boost)
    
    dst_seq = []
    for pol in pol_list:
        # --- 1. Perform baryon contraction (Up or Down Quark Insertion) ---
        if flavor == 1: # up quark insertion
            if latt_info.mpi_rank == 0:
                print(f"starting diquark contractions for up quark insertion and Polarization {pol}")
            
            src_seq = up_quark_insertion_pyquda(prop, prop, gamma_insert, PolProjections[pol])
        elif flavor == 2: # down quark insertion
            if latt_info.mpi_rank == 0:
                print(f"starting diquark contractions for down quark insertion and Polarization {pol}")
                
            src_seq = down_quark_insertion_pyquda(prop, gamma_insert, PolProjections[pol])
        else:
            raise ValueError(f"Invalid flavor: {flavor}")
        
        # --- 2. Time slicing ---
        t_source = origin[3] 
        t_sink = (t_source + t_insert) % Lt
        
        # Get data (unpacked from evenodd)
        seq_data = src_seq.lexico()
        
        # Zero out non-insertion time slices
        mask = np.zeros_like(seq_data)
        mask[t_sink, :, :, :, :, :, :, :] = 1 # only the time slice at t_sink is kept
        seq_data *= mask
        
        seq_data = evenodd(seq_data, axes=[0,1,2,3])  
        
        # --- 3. Create momentum phase ---
        # Generate phase (returns numpy or cupy usually)
        mom_phase_raw = MomentumPhase(latt_info).getPhase(momentum, x0=origin)
        mom_phase = _ensure_backend(mom_phase_raw, xp)
        
        # Get Gamma5 on correct backend
        G5 = _ensure_backend(gamma.gamma(15), xp)
        
        # Einsum contraction
        data = xp.einsum("ij, wtzyx, wtzyxkjba -> wtzyxikab", G5, mom_phase, seq_data.conj())
    
        smearing_input = core.LatticePropagator(latt_info)
        smearing_input.data = data
        
        if latt_info.mpi_rank == 0:
            print(f"diquark contractions for Polarization {pol} done")
            
        src = boosted_smearing(trafo, smearing_input, w=sm_width, boost=sm_boost)
        prop_smeared = core.invertPropagator(dirac, src, 1, 0) # NOTE or "prop_smeared = core.invertPropagator(dirac, src, 0)" depends on the quda version
        
        dst_seq.append( xp.einsum("wtzyxijfc, ik -> wtzyxjkcf", prop_smeared.data.conj(), G5) )
        
    dst_seq = _ensure_backend(dst_seq, xp)

    return dst_seq


def down_quark_insertion_pyquda(Q: LatticePropagator, Gamma, P):
    """
    PyQUDA version: Down quark insertion function (Backend Agnostic).
    """
    # --- 1. Backend & Data Prep ---
    xp = _get_xp_from_array(Q.data)
    q_data = _ensure_backend(Q.data, xp)

    original_shape = q_data.shape 
    
    # Flatten: (Vol, spin_sink, spin_src, color_sink, color_src)
    flat_Q = q_data.reshape(-1, 4, 4, 3, 3)

    # --- 2. Prepare Gamma and P matrices ---
    def to_backend_matrix(g):
        if isinstance(g, int):
            val = gamma.gamma(g)
        else:
            val = g
        return _ensure_backend(val, xp)

    G_mat = to_backend_matrix(Gamma)
    P_mat = to_backend_matrix(P)
    Gt_mat = G_mat.T 

    # --- 3. Precompute spin space matrix operations ---
    # PDu: Trace_spin(P * Q) -> Color Matrix
    PDu = xp.einsum('ij, ...jiab -> ...ab', P_mat, flat_Q)

    # GtDG: G.T * Q * G
    GtDG = xp.einsum('ij, ...jkab, kl -> ...ilab', Gt_mat, flat_Q, G_mat)

    # GtD: G.T * Q
    GtD = xp.einsum('ij, ...jkab -> ...ikab', Gt_mat, flat_Q)

    # PDG: P * Q * G
    PDG = xp.einsum('ij, ...jkab, kl -> ...ilab', P_mat, flat_Q, G_mat)

    # --- 4. Color tensor contraction ---
    eps = xp.zeros((3, 3, 3), dtype=q_data.dtype)
    # Handle item assignment for scalar/tensor backends
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[2, 1, 0] = eps[1, 0, 2] = eps[0, 2, 1] = -1

    # Term 1
    term1 = xp.einsum('abc, def, ...fc, ...uveb -> ...uvad', eps, eps, PDu, GtDG)

    # Term 2
    term2 = xp.einsum('abc, def, ...ujec, ...jkfb -> ...ukad', eps, eps, GtD, PDG)

    # Combine
    D_flat = term2 - term1

    # --- 5. Post-processing ---
    # Swap spin dimensions (axis -4 and -3)
    D_transposed = xp.swapaxes(D_flat, -4, -3)

    # Restore spacetime dimensions
    final_data = D_transposed.reshape(original_shape)

    # --- 6. Package ---
    R = core.LatticePropagator(Q.latt_info)
    R.data = final_data
    
    return R


def up_quark_insertion_pyquda(Qu: LatticePropagator, Qd: LatticePropagator, Gamma, P):
    """
    PyQUDA version: Up quark insertion function (Backend Agnostic).
    """
    # --- 1. Backend & Data Prep ---
    # Assume Qu and Qd share the same backend
    xp = _get_xp_from_array(Qu.data)
    
    qu_data = _ensure_backend(Qu.data, xp)
    qd_data = _ensure_backend(Qd.data, xp)
    
    original_shape = qu_data.shape
    # Indices: ...jkab (j=sink spin, k=src spin, a=sink color, b=src color)
    Qu_flat = qu_data.reshape(-1, 4, 4, 3, 3)
    Qd_flat = qd_data.reshape(-1, 4, 4, 3, 3)

    # --- 2. Prepare matrices ---
    def to_backend_matrix(g):
        if isinstance(g, int):
            val = gamma.gamma(g)
        else:
            val = g
        return _ensure_backend(val, xp)

    G_mat = to_backend_matrix(Gamma)
    P_mat = to_backend_matrix(P)
    Gt_mat = G_mat.T

    # --- 3. Precompute intermediate terms ---
    
    # GtDG = G.T * Qd * G
    GtDG = xp.einsum('ij, ...jkab, kl -> ...ilab', Gt_mat, Qd_flat, G_mat)

    # PDu = P * Qu
    PDu = xp.einsum('ij, ...jkab -> ...ikab', P_mat, Qu_flat)

    # DuP = Qu * P
    DuP = xp.einsum('...jkab, kl -> ...jlab', Qu_flat, P_mat)

    # TrDuP = Trace_spin(Qu * P)
    TrDuP = xp.einsum('...kjab, jk -> ...ab', Qu_flat, P_mat)

    # --- 4. Epsilon contraction ---
    eps = xp.zeros((3, 3, 3), dtype=qu_data.dtype)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1
    eps[2, 1, 0] = eps[1, 0, 2] = eps[0, 2, 1] = -1

    # Term 1: P * spin_trace(GtDG[b, e] * Du[a, d].T)
    T1_scalar = xp.einsum('...mnbe, ...mnad -> ...bead', GtDG, Qu_flat)
    R1_pre = xp.einsum('abc, def, ...bead -> ...cf', eps, eps, T1_scalar)
    R1 = xp.einsum('ij, ...cf -> ...ijcf', P_mat, R1_pre)

    # Term 2: Transpose( TrDuP[a, d] * GtDG[b, e] )
    R2 = xp.einsum('abc, def, ...ad, ...jibe -> ...ijcf', eps, eps, TrDuP, GtDG)

    # Term 3: PDu[a, d] * GtDG[b, e].T
    R3 = xp.einsum('abc, def, ...ikad, ...jkbe -> ...ijcf', eps, eps, PDu, GtDG)

    # Term 4: GtDG[a, d].T * DuP[b, e]
    R4 = xp.einsum('abc, def, ...kiad, ...klbe -> ...ilcf', eps, eps, GtDG, DuP)

    # Total Sum
    D_total = -1 * (R1 + R2 + R3 + R4)

    # --- 5. Post-processing ---
    # Swap color indices: sink, src -> src, sink
    D_final = xp.swapaxes(D_total, -1, -2)

    # Restore shape
    final_data = D_final.reshape(original_shape)

    # --- 6. Return result ---
    R = core.LatticePropagator(Qu.latt_info)
    R.data = final_data
    
    return R