"""Tests for ``scripts/precompute_forecasts.py``.

Run end-to-end in ``--replay`` mode so we stay offline, then inspect the 5
JSON files that the frontend consumes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make ``scripts`` importable without depending on pytest rootdir heuristics.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
for _p in (_SCRIPTS, _REPO_ROOT):
    sp = str(_p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from scripts import precompute_forecasts  # noqa: E402


@pytest.fixture(scope="module")
def precompute_outputs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Run the precompute pipeline once per module, writing to tmp_path."""
    out_dir = tmp_path_factory.mktemp("forecasts")
    precompute_forecasts.run(output_dir=out_dir, replay=True)
    return {
        "baseline": out_dir / "tomorrow_baseline.json",
        "heat": out_dir / "tomorrow_heat.json",
        "ev": out_dir / "tomorrow_ev.json",
        "topology": out_dir / "feeder_topology.json",
        "metrics": out_dir / "model_metrics.json",
        "generated_at": out_dir / "generated_at.json",
    }


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_all_five_files_exist(precompute_outputs: dict[str, Path]) -> None:
    for key in ("baseline", "heat", "ev", "topology", "metrics", "generated_at"):
        assert precompute_outputs[key].exists(), f"missing output: {key}"


def test_heat_peak_exceeds_baseline(precompute_outputs: dict[str, Path]) -> None:
    baseline = _read(precompute_outputs["baseline"])
    heat = _read(precompute_outputs["heat"])
    base_peak = baseline["feeder_rollup"]["peak_mw"]
    heat_peak = heat["feeder_rollup"]["peak_mw"]
    assert heat_peak >= 1.35 * base_peak, (
        f"heat peak {heat_peak:.1f} MW is not >= 1.35x baseline {base_peak:.1f} MW"
    )


def test_ev_peak_hour_in_range(precompute_outputs: dict[str, Path]) -> None:
    ev = _read(precompute_outputs["ev"])
    peak_hour = ev["feeder_rollup"]["peak_hour"]
    # The peak_hour in feeder_rollup is an index 0..23 into the 24h horizon
    # timestamps.  The EV surge injects demand only when timestamp.hour is in
    # [17, 22], so the absolute index's corresponding hour-of-day should land
    # in that window.  We read the actual hour from the quantiles row.
    quantile_rows = ev["quantiles"]
    hit_hour = quantile_rows[peak_hour]["hour"]
    assert 17 <= hit_hour <= 22, (
        f"ev peak-hour-of-day {hit_hour} (index {peak_hour}) outside [17, 22]"
    )


def test_all_132_buses_present(precompute_outputs: dict[str, Path]) -> None:
    for key in ("baseline", "heat", "ev"):
        payload = _read(precompute_outputs[key])
        assert len(payload["per_bus"]) == 132, (
            f"{key}: expected 132 per_bus entries, got {len(payload['per_bus'])}"
        )


def test_quantiles_monotonic(precompute_outputs: dict[str, Path]) -> None:
    for key in ("baseline", "heat", "ev"):
        payload = _read(precompute_outputs[key])
        rows = payload["quantiles"]
        assert len(rows) == 24, f"{key}: expected 24 quantile rows, got {len(rows)}"
        for r in rows:
            assert r["p10_mw"] <= r["p50_mw"] + 1e-6, (
                f"{key}: p10 > p50 at hour {r['hour']}: {r}"
            )
            assert r["p50_mw"] <= r["p90_mw"] + 1e-6, (
                f"{key}: p50 > p90 at hour {r['hour']}: {r}"
            )


def test_topology_payload_has_132_nodes(precompute_outputs: dict[str, Path]) -> None:
    payload = _read(precompute_outputs["topology"])
    assert payload["n_nodes"] == 132
    assert len(payload["nodes"]) == 132
    assert len(payload["edges"]) >= 100
    # coordinates normalised into [0, 1].
    for node in payload["nodes"]:
        assert 0.0 <= node["x_norm"] <= 1.0
        assert 0.0 <= node["y_norm"] <= 1.0


def test_generated_at_has_required_keys(precompute_outputs: dict[str, Path]) -> None:
    payload = _read(precompute_outputs["generated_at"])
    assert "iso" in payload
    assert payload["hours_forecast"] == 24
    assert payload["nws_source"] in {"live", "replay", "synthetic"}
    assert "git_sha" in payload
