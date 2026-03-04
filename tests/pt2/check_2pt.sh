#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPARE_PY="${ROOT_DIR}/scripts/compare_h5.py"

cd "${SCRIPT_DIR}"

python "${COMPARE_PY}" data/c2pt/S8T8_gpt_local.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 data/c2pt/S8T8_pyquda_local.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 > logs/diff_2pt.txt

python "${COMPARE_PY}" data/c2pt/S8T8_pyquda_local.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 data/c2pt/S8T8_pyquda_aurora.c2pt.0.ex.x7y3z5t7.1HYP_GSRC_W90_k3_T5.h5 > logs/diff_aurora_2pt.txt
