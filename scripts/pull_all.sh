#!/usr/bin/env bash
# GridSense-AZ — orchestrate all public-dataset pulls.
#
# Runs every puller in PLAN.md §4 order. Partial data > zero data: one failed
# puller logs a WARNING and the orchestrator keeps going. Each puller is
# wall-clock capped at 10 minutes.
#
# Flags:
#   --dry-run    Print the commands that would run, then exit 0.
#   -h | --help  Show this help text.
#
# Env / .env keys (all optional — missing keys trigger graceful SKIPs inside
# the individual pullers):
#   NREL_API_KEY   NSRDB, EVI-Pro
#   EIA_API_KEY    EIA-930
#   HF_TOKEN       EnergyBench

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
PULLER_TIMEOUT_SEC=600
PYTHON_BIN="${PYTHON_BIN:-python3}"

DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            sed -n '2,16p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "[pull_all] unknown flag: ${arg}" >&2
            exit 64
            ;;
    esac
done

# Load .env if present so keys flow through. Don't trip set -u if vars expand
# to empty; pullers handle their own key-presence checks.
if [[ -f "${REPO_ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    set -a
    . "${REPO_ROOT}/.env"
    set +a
fi

# Puller registry: "<name>|<command>" — kept in PLAN.md §4 order.
PULLERS=(
    "noaa|${PYTHON_BIN} ${SCRIPTS_DIR}/pull_noaa.py"
    "nsrdb|${PYTHON_BIN} ${SCRIPTS_DIR}/pull_nsrdb.py"
    "eia930|${PYTHON_BIN} ${SCRIPTS_DIR}/pull_eia930.py"
    "resstock|${PYTHON_BIN} ${SCRIPTS_DIR}/pull_resstock.py"
    "evi_pro|${PYTHON_BIN} ${SCRIPTS_DIR}/pull_evi_pro.py"
    "ieee123|bash ${SCRIPTS_DIR}/pull_ieee123.sh"
    "energybench|${PYTHON_BIN} ${SCRIPTS_DIR}/pull_energybench.py"
)

TOTAL=${#PULLERS[@]}

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[pull_all] --dry-run (would execute ${TOTAL} pullers)"
    for entry in "${PULLERS[@]}"; do
        name="${entry%%|*}"
        cmd="${entry#*|}"
        echo "[pull_all] would run: ${name}: timeout ${PULLER_TIMEOUT_SEC} ${cmd}"
    done
    echo "[pull_all] dry-run done | total=${TOTAL}"
    exit 0
fi

OK=0
FAIL=0
for entry in "${PULLERS[@]}"; do
    name="${entry%%|*}"
    cmd="${entry#*|}"
    echo "[pull_all] --- ${name} ---"
    # shellcheck disable=SC2086
    if timeout "${PULLER_TIMEOUT_SEC}" ${cmd}; then
        OK=$((OK + 1))
    else
        rc=$?
        echo "[pull_all] WARN: ${name} failed (exit=${rc})"
        FAIL=$((FAIL + 1))
    fi
done

echo "[pull_all] done | ok=${OK} fail=${FAIL} total=${TOTAL}"
exit 0
