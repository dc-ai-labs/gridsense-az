"""Stress-test scenarios and planner decisions for GridSense-AZ.

Composes the topology parser (:mod:`gridsense.topology`) with the OpenDSS
snapshot solver (:mod:`gridsense.power_flow`) to synthesise the two killer
what-if cases the APS brief cares about:

* a Phoenix-style **heat wave** — every loaded bus uniformly scaled up, and
* an **EV evening-peak surge** — a fixed fleet of Level-2 chargers dropped
  on a handful of residential-style buses between 6 pm and 9 pm,

plus a **combined** scenario that stacks both.

For every scenario we return a :class:`ScenarioResult` — baseline + stressed
snapshots plus derived planner-ready fields: voltage violations (ANSI C84.1
lower bound of 0.95 pu), line overloads (>100% of NormAmps), and a short
rule-based action list a distribution planner can paste into a ticket
("install capacitor at bus X", "reconductor line Y").

The scenarios are thin wrappers around
:func:`gridsense.power_flow.run_snapshot` — they only compute the right
per-bus multiplier map and then let OpenDSS do the physics.

Entry points
------------
* :func:`heat_wave_scenario`
* :func:`ev_surge_scenario`
* :func:`combined_scenario`
* :func:`rank_critical_buses`
* :func:`recommend_actions`
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable

from gridsense.power_flow import SnapshotResult, run_snapshot
from gridsense.topology import load_ieee123

__all__ = [
    "ScenarioResult",
    "heat_wave_scenario",
    "ev_surge_scenario",
    "combined_scenario",
    "rank_critical_buses",
    "recommend_actions",
    "summarise",
]


# ---------------------------------------------------------------------------
# Constants — ANSI C84.1 + rule-book thresholds
# ---------------------------------------------------------------------------

VOLTAGE_LOWER_PU: float = 0.95
"""ANSI C84.1 Range A lower service-voltage limit (per unit)."""

LINE_LOADING_LIMIT_PCT: float = 100.0
"""Normal ampacity limit — anything above is a thermal overload."""

HEAVY_VIOLATION_THRESHOLD: int = 5
"""More than this many voltage violations promotes DR advice to a
concrete capacitor-install recommendation."""

_DEFAULT_EV_TARGET_COUNT: int = 20
_DEFAULT_EV_TARGET_SEED: int = 42


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioResult:
    """Bundled output of a single stress-test run.

    Attributes:
        name: Human-readable scenario label (``"heat_wave_1.4x"``,
            ``"ev_surge_2000"``, ``"combined_1.3x_1500ev"``, …).
        baseline: Unstressed snapshot for reference (nominal IEEE 123 loads).
        stressed: Snapshot under the scenario's load overrides.
        violations: Bus names whose stressed voltage fell below 0.95 pu.
        overloads: Line names whose stressed loading exceeded 100%.
        worst_voltage_pu: Minimum voltage across all buses in the stressed run.
        worst_loading_pct: Maximum line loading across the stressed run.
        recommended_actions: Human-readable planner guidance derived from
            the stressed snapshot — see :func:`recommend_actions`.
    """

    name: str
    baseline: SnapshotResult
    stressed: SnapshotResult
    violations: list[str] = field(default_factory=list)
    overloads: list[str] = field(default_factory=list)
    worst_voltage_pu: float = 1.0
    worst_loading_pct: float = 0.0
    recommended_actions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _loaded_buses() -> dict[str, float]:
    """Return ``{bus_name: kw_load}`` for every bus with nonzero nominal load.

    Uses the pure-Python topology parser (not OpenDSS) so we can enumerate
    targets without paying a second snapshot solve.
    """
    graph = load_ieee123()
    return {
        name: float(attrs.get("kw_load", 0.0))
        for name, attrs in graph.nodes(data=True)
        if float(attrs.get("kw_load", 0.0)) > 0.0
    }


def _derive_fields(
    stressed: SnapshotResult,
) -> tuple[list[str], list[str], float, float]:
    """Pull violations, overloads, and worst-case scalars from a stressed run."""
    violations = sorted(
        bus for bus, v in stressed.bus_voltages_pu.items() if v < VOLTAGE_LOWER_PU
    )
    overloads = sorted(
        line
        for line, pct in stressed.line_loadings_pct.items()
        if pct > LINE_LOADING_LIMIT_PCT
    )
    worst_voltage = (
        min(stressed.bus_voltages_pu.values()) if stressed.bus_voltages_pu else 1.0
    )
    worst_loading = (
        max(stressed.line_loadings_pct.values()) if stressed.line_loadings_pct else 0.0
    )
    return violations, overloads, float(worst_voltage), float(worst_loading)


def _assemble(
    name: str, baseline: SnapshotResult, stressed: SnapshotResult
) -> ScenarioResult:
    """Build a final :class:`ScenarioResult` (with recommendations)."""
    violations, overloads, worst_v, worst_load = _derive_fields(stressed)
    # Build a provisional scenario so recommend_actions() can introspect it,
    # then rebuild with the finalised action list (dataclass is frozen).
    provisional = ScenarioResult(
        name=name,
        baseline=baseline,
        stressed=stressed,
        violations=violations,
        overloads=overloads,
        worst_voltage_pu=worst_v,
        worst_loading_pct=worst_load,
        recommended_actions=[],
    )
    actions = recommend_actions(provisional)
    return ScenarioResult(
        name=name,
        baseline=baseline,
        stressed=stressed,
        violations=violations,
        overloads=overloads,
        worst_voltage_pu=worst_v,
        worst_loading_pct=worst_load,
        recommended_actions=actions,
    )


def _scale_for_added_kw(existing_kw: float, added_kw: float) -> float:
    """Convert an additive kW increment into the solver's multiplier form.

    :func:`gridsense.power_flow.run_snapshot` only understands *scales* on
    existing loads.  An additive EV injection of ``added_kw`` on a bus whose
    nominal load is ``existing_kw`` therefore becomes
    ``(existing_kw + added_kw) / existing_kw``.
    """
    if existing_kw <= 0.0:
        raise ValueError("Cannot scale a bus with zero nominal load.")
    return (existing_kw + added_kw) / existing_kw


def _pick_residential_buses(
    loaded: dict[str, float],
    k: int,
    seed: int = _DEFAULT_EV_TARGET_SEED,
) -> list[str]:
    """Deterministically sample ``k`` loaded buses for EV injection.

    The IEEE 123 feeder is uniformly residential-flavoured — every loaded
    bus is a plausible L2-charger target — so we just seed :mod:`random`
    for reproducibility instead of classifying buses.
    """
    names = sorted(loaded.keys())
    k = min(k, len(names))
    rng = random.Random(seed)
    return sorted(rng.sample(names, k))


# ---------------------------------------------------------------------------
# Public API — scenarios
# ---------------------------------------------------------------------------


def heat_wave_scenario(demand_multiplier: float = 1.4) -> ScenarioResult:
    """Simulate an Arizona heat wave by scaling every loaded bus uniformly.

    A ``1.4x`` multiplier is the APS planning rule-of-thumb for a
    ``+10 °F`` day relative to design conditions; ``2.0x`` stresses the
    grid past the ANSI Range A envelope.

    Args:
        demand_multiplier: Factor applied to ``kW`` and ``kvar`` on every
            bus with nonzero nominal load.

    Returns:
        A :class:`ScenarioResult` with baseline/stressed snapshots and
        planner guidance.
    """
    if demand_multiplier <= 0.0:
        raise ValueError(f"demand_multiplier must be > 0, got {demand_multiplier}")

    loaded = _loaded_buses()
    overrides = {bus: float(demand_multiplier) for bus in loaded}

    baseline = run_snapshot()
    stressed = run_snapshot(overrides=overrides)
    return _assemble(f"heat_wave_{demand_multiplier:.1f}x", baseline, stressed)


def ev_surge_scenario(
    ev_fleet_size: int = 2000,
    target_buses: list[str] | None = None,
    kw_per_ev: float = 7.2,
) -> ScenarioResult:
    """Simulate a 6 pm-9 pm EV evening-peak injection.

    ``ev_fleet_size * kw_per_ev`` kW is split evenly across the chosen
    target buses (default: 20 randomly-selected residential-style buses
    with a fixed seed for reproducibility).

    Args:
        ev_fleet_size: Number of Level-2 chargers simultaneously online.
        target_buses: Explicit list of bus names to drop the fleet on.
            Any bus without a nominal load is dropped (the snapshot solver
            scales existing load — it can't create new load objects).
            Defaults to 20 randomly-picked loaded buses.
        kw_per_ev: Per-charger draw; ``7.2 kW`` matches an APS Smart EV
            Rewards L2 charger at 240 V / 30 A.

    Returns:
        A :class:`ScenarioResult` with baseline/stressed snapshots and
        planner guidance.
    """
    if ev_fleet_size < 0:
        raise ValueError(f"ev_fleet_size must be >= 0, got {ev_fleet_size}")
    if kw_per_ev <= 0.0:
        raise ValueError(f"kw_per_ev must be > 0, got {kw_per_ev}")

    loaded = _loaded_buses()
    if target_buses is None:
        targets: list[str] = _pick_residential_buses(loaded, k=_DEFAULT_EV_TARGET_COUNT)
    else:
        # Only keep user-supplied buses that actually have nonzero base load —
        # the snapshot solver multiplies existing kW, so zero-load buses can't
        # be nudged additively.
        targets = [b for b in target_buses if b in loaded]
        if not targets:
            raise ValueError(
                "target_buses contains no buses with nonzero nominal load"
            )

    total_added_kw = ev_fleet_size * kw_per_ev
    added_per_bus = total_added_kw / len(targets) if targets else 0.0
    overrides = {
        bus: _scale_for_added_kw(loaded[bus], added_per_bus) for bus in targets
    }

    baseline = run_snapshot()
    stressed = run_snapshot(overrides=overrides)
    return _assemble(f"ev_surge_{ev_fleet_size}", baseline, stressed)


def combined_scenario(
    demand_multiplier: float = 1.3, ev_fleet_size: int = 1500
) -> ScenarioResult:
    """Stack a heat wave on top of an EV evening peak.

    Every loaded bus is first scaled by ``demand_multiplier``; then the EV
    fleet is layered on top of the 20-bus target set using the same
    ``(existing + added) / existing`` conversion as
    :func:`ev_surge_scenario`.

    Args:
        demand_multiplier: Heat-wave scale applied to every loaded bus.
        ev_fleet_size: EV fleet size dropped on the target subset.

    Returns:
        A :class:`ScenarioResult` with baseline/stressed snapshots and
        planner guidance.
    """
    if demand_multiplier <= 0.0:
        raise ValueError(f"demand_multiplier must be > 0, got {demand_multiplier}")
    if ev_fleet_size < 0:
        raise ValueError(f"ev_fleet_size must be >= 0, got {ev_fleet_size}")

    loaded = _loaded_buses()
    targets = _pick_residential_buses(loaded, k=_DEFAULT_EV_TARGET_COUNT)

    total_added_kw = ev_fleet_size * 7.2  # same L2-charger assumption
    added_per_bus = total_added_kw / len(targets) if targets else 0.0

    overrides: dict[str, float] = {bus: float(demand_multiplier) for bus in loaded}
    for bus in targets:
        existing_kw = loaded[bus]
        scaled_kw = existing_kw * demand_multiplier
        # Scale up to heat-wave baseline, then add the EV fleet on top.
        overrides[bus] = (scaled_kw + added_per_bus) / existing_kw

    baseline = run_snapshot()
    stressed = run_snapshot(overrides=overrides)
    return _assemble(
        f"combined_{demand_multiplier:.1f}x_{ev_fleet_size}ev", baseline, stressed
    )


# ---------------------------------------------------------------------------
# Public API — planner decisions
# ---------------------------------------------------------------------------


def rank_critical_buses(
    scenario: ScenarioResult, top_k: int = 10
) -> list[tuple[str, float]]:
    """Return the ``top_k`` buses closest to (or below) the 0.95 pu limit.

    The second element is the *margin* to the limit — ``voltage - 0.95``.
    Negative values mean the bus is already in violation; the most-negative
    values are returned first.

    Args:
        scenario: A completed :class:`ScenarioResult`.
        top_k: Maximum number of entries to return.

    Returns:
        ``[(bus_name, voltage_margin), ...]`` sorted ascending by margin.
    """
    if top_k <= 0:
        return []
    margins = [
        (bus, float(v) - VOLTAGE_LOWER_PU)
        for bus, v in scenario.stressed.bus_voltages_pu.items()
    ]
    margins.sort(key=lambda pair: pair[1])
    return margins[:top_k]


def recommend_actions(scenario: ScenarioResult) -> list[str]:
    """Turn the stressed snapshot into rule-based planner guidance.

    Rules (each emits at most one concrete action string; a worst-case
    scenario therefore returns 3-4 items):

    1. **Heavy voltage violations** (more than
       :data:`HEAVY_VIOLATION_THRESHOLD` buses below 0.95 pu) → install
       a switched capacitor bank at the worst bus.
    2. **Light voltage violations** (1-threshold buses below 0.95 pu) →
       enrol the affected feeders in a TOU / pre-cooling DR programme.
    3. **Any line overload** (>100%) → reconductor the worst offender.
    4. **Marginal voltage** (min voltage between 0.95 and 0.97 pu, no
       hard violations) → proactive capacitor-bank commitment.
    5. Fallback: return an empty list — the caller decides whether to
       show "no immediate action required".

    Args:
        scenario: A :class:`ScenarioResult` with ``violations``,
            ``overloads``, and worst-case scalars already populated.

    Returns:
        Zero or more short, human-readable recommendation strings.
    """
    actions: list[str] = []

    # Rule 1 / 2 — voltage-driven recommendations.
    if len(scenario.violations) > HEAVY_VIOLATION_THRESHOLD:
        worst_bus = min(
            scenario.stressed.bus_voltages_pu.items(), key=lambda kv: kv[1]
        )[0]
        worst_pct = scenario.worst_voltage_pu * 100
        actions.append(
            f"Voltage critically low across {len(scenario.violations)} buses "
            f"(worst: bus {worst_bus} at {worst_pct:.1f}% of nominal, "
            f"minimum allowed is 95%) — install a capacitor bank near bus {worst_bus} "
            f"to restore voltage"
        )
    elif scenario.violations:
        first = ", ".join(scenario.violations[:3])
        suffix = (
            "" if len(scenario.violations) <= 3
            else f" (+{len(scenario.violations) - 3} more)"
        )
        actions.append(
            f"Voltage below safe limits at buses {first}{suffix} — "
            f"enrol nearby customers in a pre-cooling or time-of-use programme "
            f"to reduce demand before the evening peak"
        )

    # Rule 3 — thermal overloads.
    if scenario.overloads:
        worst_line = max(
            scenario.stressed.line_loadings_pct.items(), key=lambda kv: kv[1]
        )[0]
        actions.append(
            f"Line {worst_line.upper()} is carrying {scenario.worst_loading_pct:.0f}% "
            f"of its rated capacity ({len(scenario.overloads)} line(s) overloaded total) — "
            f"upgrade to a higher-rated conductor to prevent overheating"
        )

    # Rule 4 — marginal-but-not-violating voltages.
    if not scenario.violations and 0.95 <= scenario.worst_voltage_pu < 0.97:
        critical = rank_critical_buses(scenario, top_k=1)
        if critical:
            bus, _ = critical[0]
            margin_pct = (scenario.worst_voltage_pu - VOLTAGE_LOWER_PU) * 100
            actions.append(
                f"Voltage near bus {bus} is within safe limits but close to the edge "
                f"(only {margin_pct:.1f}% headroom remaining) — "
                f"consider scheduling a capacitor bank proactively before peak season"
            )

    return actions


# ---------------------------------------------------------------------------
# Convenience helpers (kept minimal — dashboard owns richer UI)
# ---------------------------------------------------------------------------


def summarise(scenarios: Iterable[ScenarioResult]) -> list[dict]:
    """Flatten a batch of scenarios into plain dicts for tabular display.

    Used by the Streamlit metrics panel — pandas can wrap the output
    directly with ``pd.DataFrame(summarise([...]))``.
    """
    rows: list[dict] = []
    for sc in scenarios:
        rows.append(
            {
                "scenario": sc.name,
                "baseline_losses_kw": sc.baseline.total_losses_kw,
                "stressed_losses_kw": sc.stressed.total_losses_kw,
                "delta_losses_kw": sc.stressed.total_losses_kw
                - sc.baseline.total_losses_kw,
                "violations": len(sc.violations),
                "overloads": len(sc.overloads),
                "worst_voltage_pu": sc.worst_voltage_pu,
                "worst_loading_pct": sc.worst_loading_pct,
                "recommendations": len(sc.recommended_actions),
            }
        )
    return rows
