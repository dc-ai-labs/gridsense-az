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
    # feeder_rollup.peak_hour is a Phoenix-local hour 0..23 (matches the
    # quantiles[].hour convention and the frontend validator's [17, 22] check).
    peak_hour = ev["feeder_rollup"]["peak_hour"]
    assert 17 <= peak_hour <= 22, (
        f"ev peak_hour {peak_hour} outside [17, 22]"
    )


def test_heat_peak_hour_in_afternoon(precompute_outputs: dict[str, Path]) -> None:
    heat = _read(precompute_outputs["heat"])
    peak_hour = heat["feeder_rollup"]["peak_hour"]
    assert 12 <= peak_hour <= 21, (
        f"heat peak_hour {peak_hour} outside afternoon heat-stress window [12, 21]"
    )


def test_heat_scenario_temp_shifted(precompute_outputs: dict[str, Path]) -> None:
    baseline = _read(precompute_outputs["baseline"])
    heat = _read(precompute_outputs["heat"])
    base_temp_f = baseline["weather"]["peak_temp_f"]
    heat_temp_f = heat["weather"]["peak_temp_f"]
    assert heat_temp_f >= base_temp_f + 9.0, (
        f"heat peak_temp_f ({heat_temp_f:.1f} F) not shifted >=9 F above baseline "
        f"({base_temp_f:.1f} F)"
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
