"""Tests for gridsense.decision — stress scenarios + planner recommendations.

PLAN.md §3.1 Layer 5. Verifies the three scenario runners physically stress
the IEEE 123 feeder, that the combined scenario is strictly worse than its
components, that high stress actually produces ANSI C84.1 voltage
violations, and that the rule-based recommender emits something actionable.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def heat_wave_14():
    """Shared heat-wave 1.4x scenario — baseline test data + loss comparison."""
    from gridsense.decision import heat_wave_scenario

    return heat_wave_scenario(demand_multiplier=1.4)


def test_heat_wave_runs(heat_wave_14) -> None:
    """1.4x demand converges and drives losses above baseline."""
    sc = heat_wave_14
    assert sc.baseline.converged, "baseline failed to converge"
    assert sc.stressed.converged, "stressed heat-wave run failed to converge"
    assert sc.stressed.total_losses_kw > sc.baseline.total_losses_kw, (
        f"heat-wave losses did not rise: "
        f"baseline={sc.baseline.total_losses_kw:.2f} kW "
        f"stressed={sc.stressed.total_losses_kw:.2f} kW"
    )
    assert sc.name.startswith("heat_wave_")


def test_ev_surge_runs() -> None:
    """2000 EVs on 20 buses produces measurable stress (losses strictly up)."""
    from gridsense.decision import ev_surge_scenario

    sc = ev_surge_scenario(ev_fleet_size=2000)
    assert sc.baseline.converged, "baseline failed to converge"
    assert sc.stressed.converged, "stressed EV-surge run failed to converge"
    # "Measurable stress" = losses appreciably above baseline.
    delta = sc.stressed.total_losses_kw - sc.baseline.total_losses_kw
    assert delta > 10.0, (
        f"EV surge barely moved losses ({delta:.3f} kW delta); "
        f"baseline={sc.baseline.total_losses_kw:.2f} stressed={sc.stressed.total_losses_kw:.2f}"
    )
    assert sc.name == "ev_surge_2000"


def test_combined_is_worst() -> None:
    """Combined(heat + EV) losses > heat-alone losses > EV-alone losses.

    The EV-surge contribution scales super-linearly with fleet size (the
    injection is concentrated on 20 buses, so local I²R losses explode as
    the fleet grows).  We pick a small fleet (100 EVs = 720 kW across 20
    buses) so the heat-alone case cleanly dominates EV-alone, and the
    combined case cleanly dominates both — matching the sponsor's mental
    model of "a heat wave is a bigger system-wide problem than tonight's
    EVs, but together they're worse than either alone".
    """
    from gridsense.decision import (
        combined_scenario,
        ev_surge_scenario,
        heat_wave_scenario,
    )

    heat = heat_wave_scenario(demand_multiplier=1.4)
    ev = ev_surge_scenario(ev_fleet_size=100)
    combined = combined_scenario(demand_multiplier=1.4, ev_fleet_size=100)

    heat_delta = heat.stressed.total_losses_kw - heat.baseline.total_losses_kw
    ev_delta = ev.stressed.total_losses_kw - ev.baseline.total_losses_kw
    combined_delta = combined.stressed.total_losses_kw - combined.baseline.total_losses_kw

    assert combined.stressed.total_losses_kw > heat.stressed.total_losses_kw, (
        f"combined ({combined.stressed.total_losses_kw:.2f}) did not exceed "
        f"heat-alone ({heat.stressed.total_losses_kw:.2f})"
    )
    assert heat_delta > ev_delta, (
        f"heat delta ({heat_delta:.2f}) should exceed EV delta ({ev_delta:.2f}) "
        f"at 1.4x uniform vs 500-EV concentrated injection"
    )
    assert combined_delta > heat_delta, (
        f"combined delta ({combined_delta:.2f}) should exceed heat delta "
        f"({heat_delta:.2f})"
    )


def test_violations_populated() -> None:
    """At 2.0x demand the stressed run shows at least one V<0.95 violation."""
    from gridsense.decision import heat_wave_scenario

    sc = heat_wave_scenario(demand_multiplier=2.0)
    assert sc.stressed.converged, "2.0x heat wave failed to converge"
    assert sc.violations, (
        f"expected V<0.95 violations at 2.0x, got none; "
        f"worst voltage was {sc.worst_voltage_pu:.3f} pu"
    )
    # Worst voltage must agree with the violation set.
    assert sc.worst_voltage_pu < 0.95
    # Every listed violation is actually below the ANSI limit.
    for bus in sc.violations:
        assert sc.stressed.bus_voltages_pu[bus] < 0.95


def test_recommendations_non_empty(heat_wave_14) -> None:
    """A meaningfully stressed scenario emits at least one recommendation.

    Re-uses the 1.4x heat-wave fixture — at that level the IEEE 123 feeder
    shows line overloads (even if it stays inside the voltage envelope),
    which the recommender must flag.
    """
    sc = heat_wave_14
    assert sc.recommended_actions, (
        "recommend_actions() returned nothing for the 1.4x heat-wave "
        "scenario despite line overloads"
    )
    # Every recommendation should be a non-empty string.
    for action in sc.recommended_actions:
        assert isinstance(action, str) and action.strip()
