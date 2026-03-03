# PyQUDA_qTMD

A Python-based framework for computing quasi Transverse Momentum Dependent distributions (qTMDs) on the lattice using [PyQUDA](https://github.com/CLQCD/PyQUDA).

## Project Structure

```
PyQUDA_qTMD/
├── utils/                  # Core utility functions for TMD calculations
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

### `Frontier/`

`Pyquda_proton_tmd_p7_T5_ts10.py` is the original production script from the Frontier cluster (AMD GPU). However, it contains several issues:

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

See [tests/README.md](tests/README.md) for detailed test descriptions and results.

## Requirements

- [PyQUDA](https://github.com/CLQCD/PyQUDA)
- [GPT](https://github.com/lehner/gpt) (for validation tests)
- cupy (NVIDIA GPU) or dpnp (Intel GPU)
- numpy, h5py
