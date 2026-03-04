#!/usr/bin/env bash
set -euo pipefail

#* TEST: run the same code with 1 GPU on local machine (NVIDIA & cupy) and Aurora (Intel & dpnp)
#* The data is generated from gpt_local.py, pyquda_local.py, and pyquda_aurora.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPARE_PY="${ROOT_DIR}/scripts/compare_h5.py"

cd "${SCRIPT_DIR}"

python "${COMPARE_PY}" data/c2pt/S8T8_pyquda_aurora.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 data/c2pt/S8T8_pyquda_local.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 > logs/pyq_aurora_vs_pyq_local_2pt.txt

python "${COMPARE_PY}" data/qTMD/S8T8_pyquda_aurora.qTMD.0.CG.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 data/qTMD/S8T8_pyquda_local.qTMD.0.CG.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 > logs/pyq_aurora_vs_pyq_local_3pt_CG.txt

python "${COMPARE_PY}" data/qTMD/S8T8_pyquda_aurora.qTMD.0.GI_PDF.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 data/qTMD/S8T8_pyquda_local.qTMD.0.GI_PDF.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 > logs/pyq_aurora_vs_pyq_local_3pt_GI.txt

python compare_npy.py data/propag/S8T8_gpt_local_srcDp.npy data/propag/S8T8_pyquda_local_srcDp.npy > logs/gpt_local_vs_pyq_local_srcDp.txt

python compare_npy.py data/propag/S8T8_gpt_local_propag_bsm.npy data/propag/S8T8_pyquda_local_propag_bsm.npy > logs/gpt_local_vs_pyq_local_propag_bsm.txt

python "${COMPARE_PY}" data/c2pt/S8T8_gpt_local.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 data/c2pt/S8T8_pyquda_local.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 > logs/gpt_local_vs_pyq_local_2pt.txt

python "${COMPARE_PY}" data/qTMD/S8T8_gpt_local.qTMD.0.CG.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 data/qTMD/S8T8_pyquda_local.qTMD.0.CG.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 > logs/gpt_local_vs_pyq_local_3pt_CG.txt

python "${COMPARE_PY}" data/qTMD/S8T8_gpt_local.qTMD.0.GI_PDF.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 data/qTMD/S8T8_pyquda_local.qTMD.0.GI_PDF.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 > logs/gpt_local_vs_pyq_local_3pt_GI.txt
