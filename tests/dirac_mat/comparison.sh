#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

python3 compare_npy.py output/S8T8_local_np_mat_noise.npy output/S8T8_local_mat_noise.npy > log/diff_local_np_mat.npy
python3 compare_npy.py output/S8T8_aurora_mat_noise.npy output/S8T8_local_mat_noise.npy > log/diff_aurora_mat.npy
python3 compare_npy.py output/S8T8_aurora_np_mat_noise.npy output/S8T8_local_mat_noise.npy > log/diff_aurora_np_mat.npy
python3 compare_npy.py output/S8T8_aurora_mpi_mat_noise.npy output/S8T8_local_mat_noise.npy > log/diff_aurora_mpi_mat.npy
python3 compare_npy.py output/S8T8_aurora_mpi_np_mat_noise.npy output/S8T8_local_mat_noise.npy > log/diff_aurora_mpi_np_mat.npy

python3 compare_npy.py output/S8T8_local_np_inv_noise.npy output/S8T8_local_inv_noise.npy > log/diff_local_np_inv.npy
python3 compare_npy.py output/S8T8_aurora_inv_noise.npy output/S8T8_local_inv_noise.npy > log/diff_aurora_inv.npy
python3 compare_npy.py output/S8T8_aurora_np_inv_noise.npy output/S8T8_local_inv_noise.npy > log/diff_aurora_np_inv.npy
python3 compare_npy.py output/S8T8_aurora_mpi_inv_noise.npy output/S8T8_local_inv_noise.npy > log/diff_aurora_mpi_inv.npy
python3 compare_npy.py output/S8T8_aurora_mpi_np_inv_noise.npy output/S8T8_local_inv_noise.npy > log/diff_aurora_mpi_np_inv.npy
