#!/usr/bin/env bash
set -euo pipefail

#* Compare results of pycontract and pyquda einsum

# source /lustre/orion/nph158/proj-shared/jinchen/env/pyq_test_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPARE_PY="${ROOT_DIR}/scripts/compare_h5.py"

cd "${SCRIPT_DIR}"

python "${COMPARE_PY}" data/c2pt/S32T48_pyquda_frontier.c2pt.300.ex.x19y23z25t35.1HYP_GSRC_W90_k3_T5.h5 data/c2pt/S32T48_nobaryon_frontier.c2pt.300.ex.x19y23z25t35.1HYP_GSRC_W90_k3_T5.h5 > tmp/pyp_vs_nobaryon_2pt_src1.txt

python "${COMPARE_PY}" data/c2pt/S32T48_pyquda_frontier.c2pt.300.ex.x19y23z25t47.1HYP_GSRC_W90_k3_T5.h5 data/c2pt/S32T48_nobaryon_frontier.c2pt.300.ex.x19y23z25t47.1HYP_GSRC_W90_k3_T5.h5 > tmp/pyp_vs_nobaryon_2pt_src2.txt

python "${COMPARE_PY}" data/qTMD/S32T48_pyquda_frontier.qTMD.300.CG.D.ex.x19y23z25t35.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 data/qTMD/S32T48_nobaryon_frontier.qTMD.300.CG.D.ex.x19y23z25t35.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 > tmp/pyp_vs_nobaryon_cg_D.txt

python "${COMPARE_PY}" data/qTMD/S32T48_pyquda_frontier.qTMD.300.CG.U.ex.x19y23z25t35.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 data/qTMD/S32T48_nobaryon_frontier.qTMD.300.CG.U.ex.x19y23z25t35.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 > tmp/pyp_vs_nobaryon_cg_U.txt

python "${COMPARE_PY}" data/qTMD/S32T48_pyquda_frontier.qTMD.300.GI_PDF.D.ex.x19y23z25t35.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 data/qTMD/S32T48_nobaryon_frontier.qTMD.300.GI_PDF.D.ex.x19y23z25t35.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 > tmp/pyp_vs_nobaryon_gi_pdf_D.txt

python "${COMPARE_PY}" data/qTMD/S32T48_pyquda_frontier.qTMD.300.GI_PDF.U.ex.x19y23z25t35.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 data/qTMD/S32T48_nobaryon_frontier.qTMD.300.GI_PDF.U.ex.x19y23z25t35.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt10.PpUnpol.5.h5 > tmp/pyp_vs_nobaryon_gi_pdf_U.txt
