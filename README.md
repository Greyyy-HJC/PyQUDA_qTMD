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
| `proton_qTMD_pyquda.py` | Proton TMD measurement class and contractions |
| `io_corr.py` | I/O utilities for correlator data (HDF5 format) |
| `tools.py` | General helper functions (source positions, MPI utilities, etc.) |

## Tests

Validation tests ensure numerical consistency between:
- **GPT** (Grid Python Toolkit) reference implementation
- **PyQUDA** with cupy backend (NVIDIA GPU)
- **PyQUDA** with dpnp backend (Intel GPU)

See [tests/README.md](tests/README.md) for detailed test descriptions and results.

## Requirements

- [PyQUDA](https://github.com/CLQCD/PyQUDA)
- [GPT](https://github.com/lehner/gpt) (for validation tests)
- cupy (NVIDIA GPU) or dpnp (Intel GPU)
- numpy, h5py
