#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

rm -f data/c2pt/*
rm -f data/qTMD/*
rm -f data/propag/*
