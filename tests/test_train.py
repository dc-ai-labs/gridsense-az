"""End-to-end smoke test for ``scripts/train.py``.

Runs the full pipeline on ~2 weeks of real AZ data with ``epochs=1`` and a
temp output dir. Budget: <60s on CPU. Asserts ``metrics.json`` exists and
contains the contract keys documented in the task brief.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make ``scripts/`` importable as a package-less module path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
for _p in (_SCRIPTS, _REPO_ROOT):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import train as train_script  # type: ignore[import-not-found]  # noqa: E402

_REQUIRED_KEYS = {"train_mae", "val_mae", "test_mae", "baseline_mae", "improvement_pct"}


@pytest.mark.slow
def test_train_smoke(tmp_path: Path) -> None:
    """Full end-to-end: 2 weeks of real data, 1 epoch, budget <60s."""
    out_dir = tmp_path / "models"
    metrics = train_script.run(
        start="2023-06-01",
        end="2023-06-15",
        epochs=1,
        device="cpu",
        out_dir=str(out_dir),
    )
    # Returned metrics dict satisfies the contract.
    assert _REQUIRED_KEYS.issubset(metrics.keys()), (
        f"missing keys: {_REQUIRED_KEYS - set(metrics.keys())}"
    )
    # And so does the on-disk copy.
    metrics_path = out_dir / "metrics.json"
    assert metrics_path.exists(), f"metrics.json not written: {metrics_path}"
    on_disk = json.loads(metrics_path.read_text())
    assert _REQUIRED_KEYS.issubset(on_disk.keys())
    # Checkpoint + history are also persisted.
    assert (out_dir / "gwnet_v0.pt").exists()
    assert (out_dir / "history.json").exists()
    # Values are finite (no NaN/Inf from a bad run).
    for k in _REQUIRED_KEYS:
        v = on_disk[k]
        assert isinstance(v, (int, float))
        # improvement_pct may legitimately be nan-ish on 1 epoch; others must be finite.
        if k != "improvement_pct":
            assert v == v and abs(v) != float("inf"), f"{k} not finite: {v}"
