# PyQUDA_qTMD

A Python-based framework for computing quasi Transverse Momentum Dependent distributions (qTMDs) on the lattice using [PyQUDA](https://github.com/CLQCD/PyQUDA).

## Project Structure

```
PyQUDA_qTMD/
├── utils/                  # Core utility functions for TMD calculations
├── scripts/                # Shared helper scripts (e.g. HDF5 comparison)
├── example_scripts/        # Production scripts for different HPC systems
│   ├── Aurora/             # Intel GPU (SYCL/dpnp backend)
│   ├── Frontier/           # AMD GPU
│   └── LQ2/                # NVIDIA GPU (cupy backend)
├── tests/                  # Validation tests
└── test_gauge/             # Sample gauge configurations for testing
```

## Utils

The `utils/` directory contains the core functions required for full TMD calculations with PyQUDA:

| Module | Description |
|--------|-------------|
| `boosted_smearing_pyquda.py` | Boosted Gaussian smearing for propagator sources |
| `bw_seq_pyquda.py` | Backward sequential propagator construction |
| `bw_seq_pycontract.py` | PyContract-based backward sequential propagator construction |
| `proton_qTMD_pyquda.py` | Proton TMD measurement class and contractions |
| `io_corr.py` | I/O utilities for correlator data (HDF5 format) |
| `tools.py` | General helper functions (source positions, MPI utilities, etc.) |

## Example Scripts

The `example_scripts/` directory contains production scripts for different HPC systems:

Shared utilities:
- `scripts/compare_h5.py`: unified HDF5 comparison helper used by all check scripts.

### `Frontier/`

Current maintained Frontier scripts:
- `pyquda_main.py` (`einsum` baseline)
- `pycontract_nobaryon.py` (`mesonAllSinkTwoPoint` path)
- `pycontract_main.py` (`mesonAllSinkTwoPoint` + `baryonSequentialTwoPoint`)

The original Frontier production script had the following known issues (fixed in maintained scripts):

1. **Redundant parameter**: `parameters["qext_PDF"]` duplicates `parameters["qext"]` unnecessarily
2. **Redundant function argument**: `Measurement.create_PDF_Wilsonline_index_list(U[0].grid)` passes a `grid` argument that is never used
3. **Task distribution bug**: The `if rank < len(tasks):` pattern only works when the number of MPI ranks ≥ number of tasks. If there are fewer ranks than tasks (e.g., 16 ranks but 32 tasks = 16 gammas × 2 flavors), the remaining tasks are silently skipped

### `LQ2/`

Cleaned-up version for NVIDIA GPUs (cupy backend):
- `gpt_main.py`: Main script with bugs fixed and code cleaned
- `gpt_utils.py`: All utility functions consolidated into one file
- `pyquda_main.py`: PyQUDA `einsum`-based implementation
- `pycontract_main.py`: PyContract-based implementation reproducing
  `pyquda_main.py` outputs while using pycontract contraction functions

Key fixes:
- Removed redundant `qext_PDF` parameter
- Fixed function signatures (removed unused arguments)
- Fixed task distribution using round-robin: `for task_idx in range(rank, len(tasks), n_ranks):`

### `Aurora/`

`pyquda_main.py`: Equivalent script for Intel GPUs (dpnp/SYCL backend), adapted from the LQ2 version.

## Tests

Validation tests ensure numerical consistency between:
- **GPT** (Grid Python Toolkit) reference implementation
- **PyQUDA** with cupy backend (NVIDIA GPU)
- **PyQUDA** with dpnp backend (Intel GPU)
- **PyContract-based PyQUDA paths** that reproduce original `einsum`-based
  PyQUDA scripts (e.g. `tests/full_TMD/pycontract_local.py` and
  `example_scripts/LQ2/pycontract_main.py`)

### Test Suites

- `tests/dirac_mat/`:
  validates `dirac.mat()` consistency across `numpy`, `cupy`, and `dpnp`.
- `tests/pt2/`:
  validates the 2-point pipeline (boosted smearing, inversion, and 2pt contraction).
- `tests/full_TMD/`:
  validates full TMD observables (2pt/3pt, CG TMD, GI PDF), including
  `pycontract_local.py` as the pycontract replacement path for the original
  `einsum`-based contractions.

### PyContract Equivalence Workflow (Local Test)

Reference scripts:
- `tests/full_TMD/pyquda_local.py`
- `tests/full_TMD/pycontract_local.py`
- `tests/full_TMD/check_pycontract.sh`

Typical workflow:

```bash
cd /home/jinchen/git/lat-software/PyQUDA_qTMD
source /home/jinchen/miniconda3/etc/profile.d/conda.sh
conda activate pygpt

python -m tests.full_TMD.pyquda_local
python -m tests.full_TMD.pycontract_local
bash tests/full_TMD/check_pycontract.sh
```

### Important Numerical Notes

To align GPT/PyQUDA behavior and improve reproducibility, the following
modifications were used in validation:

1. **PyQUDA**: FFT definition in boosted smearing was adjusted to match GPT convention.
2. **GPT** (`gpt/lib/cgpt/lib/coordinates.cc`, line 488): `ComplexF` -> `ComplexD`.
3. **GPT** (`gpt/lib/cgpt/lib/coordinates.cc`, line 454): higher-precision `pi`.

`dirac.setPrecision(sloppy=8)` helps reduce GPT-vs-PyQUDA differences by making
iteration paths more consistent, but the distance to the exact solution is still
set by the solver tolerance.

### Current Test Results

#### S8T8 Configuration

Comparing `gpt_local`, `pyquda_local`, and `pyquda_aurora`:

| Quantity | Rel. Diff (tol=1e-10) | Rel. Diff (tol=1e-12) |
|----------|----------------------|----------------------|
| Boosted smeared propagator | ~1e-8 | ~1e-11 |
| 2-point correlator | ~1e-9 | ~1e-9 |
| 3-point correlator (CG & GI) | ~1e-9 | ~1e-11 |

Detailed logs: `tests/full_TMD/logs/`.

#### S64T64 Production Configuration

Comparing LQ2 `gpt_main.py` (NVIDIA) vs Aurora `pyquda_main.py` (Intel):

| Quantity | Rel. Diff (tol=1e-10) | Rel. Diff (tol=1e-12) |
|----------|----------------------|----------------------|
| 2-point correlator | ~1e-8 | ~1e-10 |
| 3-point correlator (CG) | ~1e-7 | ~1e-9 |
| 3-point correlator (GI) | ~1e-8 | ~1e-10 |

Detailed logs: `example_scripts/logs/`.

## PyContract Results on LQ2 and Frontier

Frontier validation artifacts are in:
- scripts: `example_scripts/Frontier/pyquda_main.py`, `example_scripts/Frontier/pycontract_nobaryon.py`, `example_scripts/Frontier/pycontract_main.py`
- logs: `example_scripts/Frontier/logs/nobaryon.4172378.out`, `example_scripts/Frontier/logs/pyquda.4172381.out`, `example_scripts/Frontier/logs/pycontract.4172499.out`
- HDF5 comparisons: `example_scripts/Frontier/check_nobaryon.sh`, `example_scripts/Frontier/check_pycontract.sh`, shared tool `scripts/compare_h5.py`, and outputs under `example_scripts/Frontier/tmp/`

Observed results:
- **Frontier, meson-only pycontract path** (`mesonAllSinkTwoPoint`, via `pycontract_nobaryon.py`):
  - Compared to original `einsum` output, **2pt / CG TMD / GI PDF** are consistent.
  - Using the current production criterion, the maximum relative differences are at the order of **~100x inversion tolerance** (and remain within expected numerical precision for these runs).
  - Contraction is significantly faster:
    - source 1: `contract_TMD over` = `5.23s` (nobaryon) vs `30.90s` (einsum), about `5.9x`
    - source 2: `contract_TMD over` = `3.24s` (nobaryon) vs `29.18s` (einsum), about `9.0x`
    - effective speedup is roughly **7-8x**.
- **Full pycontract path** (`mesonAllSinkTwoPoint` + `baryonSequentialTwoPoint`, via `pycontract_main.py`):
  - On **LQ2** (NVIDIA), **2pt / CG TMD / GI PDF** remain consistent with the original `einsum` pipeline.
  - On **Frontier** (AMD), **D flavor** remains consistent, but **U flavor** shows mismatch in **CG TMD** and **GI PDF** compared with original `einsum`.
  - This behavior is reflected by large U-flavor relative differences in the Frontier comparison outputs (`tmp/pyp_vs_pycontract_cg_U.txt`, `tmp/pyp_vs_pycontract_gi_pdf_U.txt`).

Current practical recommendation:
- On Frontier, use the meson-only pycontract path (`pycontract_nobaryon.py`) for production comparisons.
- Keep `baryonSequentialTwoPoint` on Frontier under investigation before enabling it in production.

## Requirements

- [PyQUDA](https://github.com/CLQCD/PyQUDA)
- [GPT](https://github.com/lehner/gpt) (for validation tests)
- cupy (NVIDIA GPU) or dpnp (Intel GPU)
- numpy, h5py
