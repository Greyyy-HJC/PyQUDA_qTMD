#* TEST: run the same code with 1 GPU on local machine (NVIDIA & cupy) and Aurora (Intel & dpnp)
#* The data is generated from check_local.py and check_aurora.py

python compare_h5.py data/c2pt/S8T8_aurora.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 data/c2pt/S8T8_local.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 > logs/diff_2pt.txt

python compare_h5.py data/qTMD/S8T8_aurora.qTMD.0.CG.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 data/qTMD/S8T8_local.qTMD.0.CG.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 > logs/diff_3pt_CG.txt

python compare_h5.py data/qTMD/S8T8_aurora.qTMD.0.GI_PDF.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.h5 data/qTMD/S8T8_local.qTMD.0.GI_PDF.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.h5 > logs/diff_3pt_GI.txt




python compare_h5.py data/c2pt/S8T8_aurora_mpi.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 data/c2pt/S8T8_local.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 > logs/diff_2pt_mpi.txt

python compare_h5.py data/qTMD/S8T8_aurora_mpi.qTMD.0.CG.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 data/qTMD/S8T8_local.qTMD.0.CG.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 > logs/diff_3pt_CG_mpi.txt

python compare_h5.py data/qTMD/S8T8_aurora_mpi.qTMD.0.GI_PDF.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.h5 data/qTMD/S8T8_local.qTMD.0.GI_PDF.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.h5 > logs/diff_3pt_GI_mpi.txt