python3 compare_txt.py S8T8_local_np_inv_point_src.txt S8T8_local_inv_point_src.txt > diff_local_np.txt
python3 compare_txt.py S8T8_aurora_inv_point_src.txt S8T8_local_inv_point_src.txt > diff_aurora.txt
python3 compare_txt.py S8T8_aurora_mpi_inv_point_src.txt S8T8_local_inv_point_src.txt > diff_aurora_mpi.txt
python3 compare_txt.py S8T8_aurora_mpi_np_inv_point_src.txt S8T8_local_inv_point_src.txt > diff_aurora_mpi_np.txt