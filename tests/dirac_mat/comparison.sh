python3 compare_npy.py output/S8T8_local_np_mat_noise.npy output/S8T8_local_mat_noise.npy > log/diff_local_np.npy
python3 compare_npy.py output/S8T8_aurora_mat_noise.npy output/S8T8_local_mat_noise.npy > log/diff_aurora.npy
python3 compare_npy.py output/S8T8_aurora_np_mat_noise.npy output/S8T8_local_mat_noise.npy > log/diff_aurora_np.npy
python3 compare_npy.py output/S8T8_aurora_mpi_mat_noise.npy output/S8T8_local_mat_noise.npy > log/diff_aurora_mpi.npy
python3 compare_npy.py output/S8T8_aurora_mpi_np_mat_noise.npy output/S8T8_local_mat_noise.npy > log/diff_aurora_mpi_np.npy