python compare_npy.py data/propag/S8T8_pyquda_local_propag_bsm.npy data/propag/S8T8_gpt_local_propag_bsm.npy > logs/diff_propag_bsm.txt

python compare_npy.py data/propag/S8T8_pyquda_local_srcDp.npy data/propag/S8T8_gpt_local_srcDp.npy > logs/diff_srcDp.txt

python compare_npy.py data/propag/S8T8_pyquda_aurora_propag_bsm.npy data/propag/S8T8_pyquda_local_propag_bsm.npy > logs/diff_aurora_propag_bsm.txt

python compare_npy.py data/propag/S8T8_pyquda_aurora_srcDp.npy data/propag/S8T8_pyquda_local_srcDp.npy > logs/diff_aurora_srcDp.txt