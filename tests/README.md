# Tests for PyQUDA qTMD

This directory contains validation tests comparing GPT (Grid Python Toolkit) and PyQUDA implementations across different backends (local CUDA via cupy, Aurora Intel GPU via dpnp).

## Test Suites

### `dirac_mat/`
Tests the Dirac matrix operations (`dirac.mat()`) to verify that the fermion matrix application produces consistent results across different backends (numpy, cupy, dpnp).

### `pt2/`
Tests the 2-point correlation function calculation pipeline, including:
- Boosted Gaussian smearing of propagator sources
- Forward propagator inversion
- 2-point contraction

### `full_TMD/`
Full TMD (Transverse Momentum Dependent) measurement tests, including:
- Boosted smeared source and propagator
- 2-point and 3-point correlation functions
- CG (Coulomb Gauge) TMD contractions
- GI (Gauge Invariant) PDF contractions

## Code Modifications

Three modifications were made to achieve numerical consistency between GPT and PyQUDA:

1. **PyQUDA**: Changed the FFT definition in boosted smearing to match GPT's convention.

2. **GPT** (`gpt/lib/cgpt/lib/coordinates.cc` line 488): Changed `ComplexF` to `ComplexD` for higher precision in coordinate calculations.

3. **GPT** (`gpt/lib/cgpt/lib/coordinates.cc` line 454): Increased π precision from `3.14159265359` to `3.14159265358979323846`.

## Notes on `dirac.setPrecision(sloppy=8)`

Setting `dirac.setPrecision(sloppy=8)` (double precision for sloppy solves) helps reduce the difference between GPT and PyQUDA results. However, it's important to understand what this achieves:

- It makes the **iteration paths converge** between the two implementations
- This results in differences between the two solutions being **much smaller than the inversion tolerance**
- However, the **distance from the true solution** is still at the order of the specified precision (tolerance)

In other words, `setPrecision(sloppy=8)` makes the two implementations agree with each other, but both solutions still have residuals at the tolerance level relative to the exact solution.

## Current Test Results

### S8T8 Test Configuration

Comparing `gpt_local`, `pyquda_local`, and `pyquda_aurora`:

| Quantity | Rel. Diff (tol=1e-10) | Rel. Diff (tol=1e-12) |
|----------|----------------------|----------------------|
| Boosted smeared propagator | ~1e-8 | ~1e-11 |
| 2-point correlator | ~1e-9 | ~1e-9 |
| 3-point correlator (CG & GI) | ~1e-9 | ~1e-11 |

Detailed comparison logs are available in `tests/full_TMD/logs/`.

### S64T64 Production Configuration

Comparing `gpt_main.py` on LQ2 (NVIDIA GPU) vs `pyquda_main.py` on Aurora (Intel GPU):

| Quantity | Rel. Diff (tol=1e-10) |
|----------|----------------------|
| 2-point correlator | ~1e-8 |
| 3-point correlator (CG) | ~1e-7 |
| 3-point correlator (GI) | ~1e-8 |

Detailed comparison logs are available in `example_scripts/logs/`.

This validates cross-platform consistency on production-scale lattices.

**Notes on error sources:**

- **Propagator**: The inversion error is uniformly distributed across all lattice sites. Since some sites have small absolute values, their relative errors appear larger. Increasing inversion precision directly improves the relative difference.

- **2-point correlator**: The error is dominated by `einsum` summation accumulation, with an absolute error floor of ~1e-7 that cannot be further reduced by improving inversion precision.

- **3-point correlator**: The error scales with inversion precision and can be effectively reduced by using tighter tolerances.

These results demonstrate excellent agreement between all three implementations.