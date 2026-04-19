"""Tests for gridsense.power_flow — OpenDSSDirect snapshot wrapper.

PLAN.md §5.2. Verifies the snapshot solver converges on the canonical
IEEE 123 feeder, returns bus/line data aligned with the topology module's
node keys, and that load overrides scale losses without leaking state
between calls.
"""

from __future__ import annotations


def test_importable() -> None:
    """Module loads and exposes the public API."""
    import gridsense.power_flow as pf

    assert hasattr(pf, "run_snapshot")
    assert hasattr(pf, "SnapshotResult")
    assert callable(pf.run_snapshot)


def test_run_snapshot_converges() -> None:
    """Snapshot runs, converged=True, iterations > 0."""
    from gridsense.power_flow import run_snapshot

    result = run_snapshot()
    assert result.converged is True, "solver did not converge"
    assert result.iterations > 0, f"iterations={result.iterations}"
    # Sanity on the result container itself.
    assert isinstance(result.bus_voltages_pu, dict)
    assert isinstance(result.line_loadings_pct, dict)


def test_bus_voltage_shape() -> None:
    """Every topology bus appears in bus_voltages_pu with a finite value.

    The topology parser and the OpenDSS engine agree on 132 buses for the
    IEEE 123 feeder (OpenDSS's internal ``sourcebus`` sits *behind* the
    Vsource and is not in AllBusNames, so it never shows up in the topology
    graph either — no exclusion set needed).
    """
    from gridsense.power_flow import run_snapshot
    from gridsense.topology import load_ieee123

    result = run_snapshot()
    topo_buses = set(load_ieee123().nodes())
    pf_buses = set(result.bus_voltages_pu.keys())

    missing = topo_buses - pf_buses
    assert not missing, f"{len(missing)} topology buses missing voltage: {sorted(missing)[:10]}"

    for bus in topo_buses:
        v = result.bus_voltages_pu[bus]
        assert isinstance(v, float)
        assert v == v, f"NaN voltage at {bus}"  # NaN check
        assert 0.0 < v < 10.0, f"nonsensical voltage at {bus}: {v}"


def test_voltage_magnitudes_sane() -> None:
    """95% of bus voltages lie in [0.85, 1.10] pu; none below 0.5 (divergence tell)."""
    from gridsense.power_flow import run_snapshot

    result = run_snapshot()
    voltages = list(result.bus_voltages_pu.values())
    assert voltages, "no bus voltages returned"

    in_band = sum(1 for v in voltages if 0.85 <= v <= 1.10)
    pct = in_band / len(voltages)
    assert pct >= 0.95, (
        f"only {in_band}/{len(voltages)} ({pct:.1%}) buses in [0.85, 1.10] pu"
    )

    assert min(voltages) > 0.5, (
        f"min voltage {min(voltages):.3f} pu suggests divergence"
    )


def test_line_loadings_non_negative() -> None:
    """All line loading percentages are ≥ 0."""
    from gridsense.power_flow import run_snapshot

    result = run_snapshot()
    assert result.line_loadings_pct, "no line loadings returned"
    for name, pct in result.line_loadings_pct.items():
        assert pct >= 0.0, f"negative loading {pct} on line {name!r}"


def test_overrides_increase_losses() -> None:
    """Scaling load at bus 42 upward should produce strictly higher total losses."""
    from gridsense.power_flow import run_snapshot

    nominal = run_snapshot()
    stressed = run_snapshot(overrides={"42": 2.0})

    assert stressed.converged, "stressed case failed to converge"
    assert stressed.total_losses_kw > nominal.total_losses_kw, (
        f"overrides did not raise losses: "
        f"nominal={nominal.total_losses_kw:.3f}  stressed={stressed.total_losses_kw:.3f}"
    )


def test_overrides_is_pure() -> None:
    """A nominal run after an override run returns the original losses.

    Catches any state leak in the OpenDSS global engine — Clear+Redirect
    must fully reset the model between calls.
    """
    from gridsense.power_flow import run_snapshot

    baseline = run_snapshot()
    # Mutate loads aggressively to make any leak obvious.
    _ = run_snapshot(overrides={"42": 3.0, "48": 3.0, "49": 3.0, "50": 3.0})
    after = run_snapshot()

    assert abs(after.total_losses_kw - baseline.total_losses_kw) < 1e-6, (
        f"losses drifted: baseline={baseline.total_losses_kw}  after={after.total_losses_kw}"
    )
    assert abs(after.total_losses_kvar - baseline.total_losses_kvar) < 1e-6

    # Voltages should also be identical for at least one canonical bus.
    for bus in ("150", "1", "42"):
        if bus in baseline.bus_voltages_pu and bus in after.bus_voltages_pu:
            assert abs(after.bus_voltages_pu[bus] - baseline.bus_voltages_pu[bus]) < 1e-9, (
                f"voltage at {bus} drifted after override round-trip"
            )
