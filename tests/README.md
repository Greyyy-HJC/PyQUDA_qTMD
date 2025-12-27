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

| Quantity | Relative Difference |
|----------|---------------------|
| Boosted smeared propagator | ~1e-10 |
| 2-point correlator | ~1e-9 |
| 3-point correlator (CG & GI) | ~1e-9 |

These results demonstrate excellent agreement between all three implementations.

### S64T64 Production Configuration (In Progress)

A larger-scale validation test on the S64T64 configuration is currently running:

- **Aurora** (Intel GPU): Running `example_scripts/Aurora/pyquda_main.py` with dpnp backend
- **LQ2** (NVIDIA GPU): Running `example_scripts/LQ2/gpt_main.py` with cupy backend

This test will validate cross-platform consistency on production-scale lattices.

