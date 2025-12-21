python3 compare_npy.py S8T8_local_np_inv_point_src.npy S8T8_local_inv_point_src.npy > diff_local_np.npy
python3 compare_npy.py S8T8_aurora_inv_point_src.npy S8T8_local_inv_point_src.npy > diff_aurora.npy
python3 compare_npy.py S8T8_aurora_np_inv_point_src.npy S8T8_local_inv_point_src.npy > diff_aurora_np.npy
python3 compare_npy.py S8T8_aurora_mpi_inv_point_src.npy S8T8_local_inv_point_src.npy > diff_aurora_mpi.npy
python3 compare_npy.py S8T8_aurora_mpi_np_inv_point_src.npy S8T8_local_inv_point_src.npy > diff_aurora_mpi_np.npy