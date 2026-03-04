#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPARE_PY="${ROOT_DIR}/scripts/compare_h5.py"

cd "${ROOT_DIR}"

python "${COMPARE_PY}" \
tests/full_TMD/data/c2pt/S8T8_pyquda_local.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 \
tests/full_TMD/data/c2pt/S8T8_pyquda_local_pycontract.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5

python "${COMPARE_PY}" \
tests/full_TMD/data/qTMD/S8T8_pyquda_local.qTMD.0.CG.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 \
tests/full_TMD/data/qTMD/S8T8_pyquda_local_pycontract.qTMD.0.CG.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5

python "${COMPARE_PY}" \
tests/full_TMD/data/qTMD/S8T8_pyquda_local.qTMD.0.CG.U.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 \
tests/full_TMD/data/qTMD/S8T8_pyquda_local_pycontract.qTMD.0.CG.U.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5

python "${COMPARE_PY}" \
tests/full_TMD/data/qTMD/S8T8_pyquda_local.qTMD.0.GI_PDF.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 \
tests/full_TMD/data/qTMD/S8T8_pyquda_local_pycontract.qTMD.0.GI_PDF.D.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5

python "${COMPARE_PY}" \
tests/full_TMD/data/qTMD/S8T8_pyquda_local.qTMD.0.GI_PDF.U.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5 \
tests/full_TMD/data/qTMD/S8T8_pyquda_local_pycontract.qTMD.0.GI_PDF.U.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.PX0PY0PZ9dt4.PpUnpol.5.h5
