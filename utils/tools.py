#!/usr/bin/env python3
#
# GPT inversion sources selection
#
import os

# ---------- Backend Helpers (consistent with boosted_smearing_pyquda) ----------
def _get_xp_from_array(a):
    """Return the base module of the array's type, e.g. cupy / numpy / dpnp / torch."""
    # Handle case where data might be wrapped or None, default to numpy if unsure
    if a is None:
        return __import__("numpy")
    base = type(a).__module__.split('.')[0]
    return __import__(base)

def _ensure_backend(x, xp):
    """Move x to the same backend as xp if needed."""
    # Check if x is already on the correct backend
    if type(x).__module__.split('.')[0] == xp.__name__:
        return x
    # Handle transfer
    if hasattr(xp, "asarray"):
        return xp.asarray(x)
    if xp.__name__ == "torch":
        return xp.as_tensor(x)
    return xp.array(x)

# --- HELPER: Ensure new arrays are on the EXACT SAME QUEUE as the data ---
def _asarray_on_queue(val, xp, ref_arr):
    """
    Creates an array 'val' on the same backend and SYCL queue as 'ref_arr'.
    """
    # 1. If usage is dpnp, strictly enforce the sycl_queue
    if xp.__name__ == 'dpnp' and hasattr(ref_arr, 'sycl_queue'):
        return xp.asarray(val, sycl_queue=ref_arr.sycl_queue)
    
    # 2. Fallback for standard numpy/cupy or if ref_arr has no queue info
    return xp.asarray(val)

def mpi_print(latt_info, message):
    if latt_info.mpi_rank == 0:
        print(message)


def srcLoc_distri_eq(L, src_origin):
    source_positions = []
    i_src = 0
    div = 4
    for i in range(div):
        for j in range(div):
            for k in range(div):
                for l in range(div):
                    source_positions += [[round(i*L[0]/div+src_origin[0])%L[0], round(j*L[1]/div+src_origin[1])%L[1], round(k*L[2]/div+src_origin[2])%L[2], round(l*L[3]/div+src_origin[3])%L[3]]]
    return source_positions


'''
    # random source creation
    job_seed = job.split("_correlated")[0]
    rng = g.random(f"2PT-ensemble-{conf}-{job_seed}")
    source_positions_sloppy = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
        for j in range(jobs[job]["sloppy"])
    ]
    source_positions_exact = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
        for j in range(jobs[job]["exact"])
    ]
'''

