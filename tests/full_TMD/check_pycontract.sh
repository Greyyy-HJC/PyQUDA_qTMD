cd /home/jinchen/git/lat-software/PyQUDA_qTMD

python tests/full_TMD/compare_h5.py \
tests/full_TMD/data/c2pt/S8T8_pyquda_local.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 \
tests/full_TMD/data/c2pt/S8T8_pyquda_local_pycontract.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5

python tests/full_TMD/compare_h5.py \
tests/full_TMD/data/qTMD/S8T8_pyquda_local.qTMD.0.CG.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 \
tests/full_TMD/data/qTMD/S8T8_pyquda_local_pycontract.qTMD.0.CG.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5

python tests/full_TMD/compare_h5.py \
tests/full_TMD/data/qTMD/S8T8_pyquda_local.qTMD.0.CG.U.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 \
tests/full_TMD/data/qTMD/S8T8_pyquda_local_pycontract.qTMD.0.CG.U.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5

python tests/full_TMD/compare_h5.py \
tests/full_TMD/data/qTMD/S8T8_pyquda_local.qTMD.0.GI_PDF.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 \
tests/full_TMD/data/qTMD/S8T8_pyquda_local_pycontract.qTMD.0.GI_PDF.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5

python tests/full_TMD/compare_h5.py \
tests/full_TMD/data/qTMD/S8T8_pyquda_local.qTMD.0.GI_PDF.U.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 \
tests/full_TMD/data/qTMD/S8T8_pyquda_local_pycontract.qTMD.0.GI_PDF.U.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5
