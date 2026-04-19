"""Offline test suite for the dataset pullers in ``scripts/``.

No network calls, no API keys. Every test must pass in a hermetic CI.

Coverage:
    * syntactic / import validity of every puller
    * ``--help`` smoke-test for each Python puller
    * graceful-skip-on-missing-key contract for the keyed pullers
    * ``pull_all.sh --dry-run`` lists all 7 pullers
    * ``pull_resstock.py --explain`` validates SQL without network
    * ``pull_ieee123.sh`` short-circuits when the master DSS file is cached
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
IEEE123_MASTER = REPO_ROOT / "data" / "raw" / "ieee123" / "IEEE123Master.dss"

PY_PULLERS = (
    "pull_noaa.py",
    "pull_nsrdb.py",
    "pull_eia930.py",
    "pull_resstock.py",
    "pull_evi_pro.py",
    "pull_energybench.py",
)
SH_PULLERS = (
    "pull_ieee123.sh",
    "pull_all.sh",
)
KEYED_PY_PULLERS = (
    ("pull_nsrdb.py", "NREL_API_KEY"),
    ("pull_eia930.py", "EIA_API_KEY"),
    ("pull_evi_pro.py", "NREL_API_KEY"),
    ("pull_energybench.py", "HF_TOKEN"),
)
KEY_VARS_TO_CLEAR = ("NREL_API_KEY", "EIA_API_KEY", "HF_TOKEN")

SUBPROCESS_TIMEOUT = 30


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_env() -> dict[str, str]:
    """Return a ``subprocess.run``-ready env dict with all API keys stripped.

    Keeps ``PATH`` / ``HOME`` / interpreter basics — only removes the three
    puller-relevant keys so tests exercise the graceful-skip paths.
    """
    env = os.environ.copy()
    for var in KEY_VARS_TO_CLEAR:
        env.pop(var, None)
    return env


# ---------------------------------------------------------------------------
# syntactic validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", PY_PULLERS)
def test_all_pullers_importable(script: str) -> None:
    """Every Python puller must at minimum parse and load as a module spec."""
    path = SCRIPTS_DIR / script
    assert path.exists(), f"missing puller: {path}"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None, f"could not build spec for {path}"
    # ``py_compile`` parses the source without executing it — catches
    # SyntaxError without hitting any import-time network calls.
    rc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        capture_output=True,
        timeout=SUBPROCESS_TIMEOUT,
    )
    assert rc.returncode == 0, f"py_compile failed for {script}: {rc.stderr.decode()}"


# ---------------------------------------------------------------------------
# argparse --help
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script", PY_PULLERS)
def test_pullers_argparse_help(script: str) -> None:
    """Every Python puller must expose ``--help`` and exit 0."""
    path = SCRIPTS_DIR / script
    proc = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True,
        timeout=SUBPROCESS_TIMEOUT,
    )
    assert proc.returncode == 0, (
        f"{script} --help exit={proc.returncode} stderr={proc.stderr.decode()[:400]}"
    )
    assert b"usage" in proc.stdout.lower() or b"usage" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# graceful skip without keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("script,required_env", KEYED_PY_PULLERS)
def test_pullers_graceful_skip_without_keys(
    script: str, required_env: str, clean_env: dict[str, str]
) -> None:
    """With no API keys set, keyed pullers must exit 0 and announce SKIP."""
    path = SCRIPTS_DIR / script
    proc = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        timeout=SUBPROCESS_TIMEOUT,
        env=clean_env,
    )
    combined = (proc.stdout + proc.stderr).decode(errors="replace").lower()
    assert proc.returncode == 0, (
        f"{script} returncode={proc.returncode}, required_env={required_env}, "
        f"output={combined[:400]}"
    )
    assert "skip" in combined, f"{script} output missing 'skip' marker: {combined[:400]}"


# ---------------------------------------------------------------------------
# pull_all.sh --dry-run
# ---------------------------------------------------------------------------


def test_pull_all_dry_run() -> None:
    """``pull_all.sh --dry-run`` must exit 0 and list all 7 pullers by name."""
    proc = subprocess.run(
        ["bash", str(SCRIPTS_DIR / "pull_all.sh"), "--dry-run"],
        capture_output=True,
        timeout=SUBPROCESS_TIMEOUT,
    )
    out = proc.stdout.decode()
    assert proc.returncode == 0, f"pull_all.sh --dry-run failed: {proc.stderr.decode()[:400]}"
    for name in ("noaa", "nsrdb", "eia930", "resstock", "evi_pro", "ieee123", "energybench"):
        assert name in out, f"dry-run missing {name}: {out}"
    # Sanity — the final summary line reports the total.
    assert "total=7" in out, f"dry-run missing total=7: {out}"


# ---------------------------------------------------------------------------
# pull_resstock --explain
# ---------------------------------------------------------------------------


def test_pull_resstock_explain() -> None:
    """``pull_resstock.py --explain`` validates SQL via DuckDB without network."""
    duckdb_spec = importlib.util.find_spec("duckdb")
    if duckdb_spec is None:
        pytest.skip("duckdb not installed")
    path = SCRIPTS_DIR / "pull_resstock.py"
    proc = subprocess.run(
        [sys.executable, str(path), "--explain"],
        capture_output=True,
        timeout=SUBPROCESS_TIMEOUT,
    )
    combined = (proc.stdout + proc.stderr).decode(errors="replace")
    assert proc.returncode == 0, f"explain exit={proc.returncode} output={combined[:400]}"
    assert "EXPLAIN" in combined.upper() or "explain" in combined.lower()


# ---------------------------------------------------------------------------
# pull_ieee123 — skip path when already cached
# ---------------------------------------------------------------------------


def test_pull_ieee123_skip_when_cached() -> None:
    """If the IEEE 123 master DSS is present, the bash puller should exit 0 fast.

    If the sentinel is absent, the script would re-clone tshort/OpenDSS; that's
    a network op we do not run in the offline test suite — skip in that case.
    """
    if shutil.which("bash") is None:
        pytest.skip("bash not on PATH")
    if not IEEE123_MASTER.exists():
        pytest.skip(f"sentinel absent; would re-clone: {IEEE123_MASTER}")
    proc = subprocess.run(
        ["bash", str(SCRIPTS_DIR / "pull_ieee123.sh")],
        capture_output=True,
        timeout=SUBPROCESS_TIMEOUT,
    )
    combined = (proc.stdout + proc.stderr).decode(errors="replace")
    assert proc.returncode == 0, f"pull_ieee123 exit={proc.returncode} output={combined[:400]}"
    assert "skip" in combined.lower() or "cached" in combined.lower()
