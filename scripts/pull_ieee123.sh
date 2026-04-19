#!/usr/bin/env bash
# Pull IEEE 123-bus feeder .dss files from tshort/OpenDSS (public).
# PLAN.md §4.6. No auth. Writes to data/raw/ieee123/.
#
# The canonical master file inside Distrib/IEEETestCases/123Bus is `IEEE123Master.dss`
# (verified against https://github.com/tshort/OpenDSS on 2026-04-18).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${REPO_ROOT}/data/raw"
CLONE_DIR="${RAW_DIR}/opendss_repo"
TARGET_DIR="${RAW_DIR}/ieee123"
MASTER_FILE="IEEE123Master.dss"
FORCE="${FORCE:-0}"

echo "[pull_ieee123] repo=${REPO_ROOT}"
mkdir -p "${RAW_DIR}" "${TARGET_DIR}"

if [[ -f "${TARGET_DIR}/${MASTER_FILE}" ]] && [[ "${FORCE}" != "1" ]]; then
    echo "[pull_ieee123] skip — ${TARGET_DIR}/${MASTER_FILE} already exists"
    echo "[pull_ieee123] ok (cached)"
    exit 0
fi

# Fresh clone (shallow).  Remove any previous partial clone first.
if [[ -d "${CLONE_DIR}/.git" ]] && [[ "${FORCE}" != "1" ]]; then
    echo "[pull_ieee123] repo already cloned at ${CLONE_DIR} — reusing"
else
    rm -rf "${CLONE_DIR}"
    echo "[pull_ieee123] shallow-cloning tshort/OpenDSS into ${CLONE_DIR}"
    git clone --depth 1 --quiet https://github.com/tshort/OpenDSS.git "${CLONE_DIR}"
fi

SRC_123="${CLONE_DIR}/Distrib/IEEETestCases/123Bus"
if [[ ! -d "${SRC_123}" ]]; then
    echo "[pull_ieee123] ERROR: expected source dir not found: ${SRC_123}" >&2
    exit 2
fi

# Preserve case — Linux is case-sensitive; the repo stores IEEE123Master.dss.
cp -r "${SRC_123}/." "${TARGET_DIR}/"

# Sanity — canonical master file present.
if [[ ! -f "${TARGET_DIR}/${MASTER_FILE}" ]]; then
    echo "[pull_ieee123] FAIL: ${TARGET_DIR}/${MASTER_FILE} missing after copy" >&2
    echo "[pull_ieee123] contents of target dir:" >&2
    ls -la "${TARGET_DIR}" >&2 || true
    exit 3
fi

echo "[pull_ieee123] ok master=${TARGET_DIR}/${MASTER_FILE}"
echo "[pull_ieee123] file count: $(find "${TARGET_DIR}" -maxdepth 1 -type f | wc -l)"
exit 0
