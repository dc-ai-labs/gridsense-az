"""OpenDSSDirect snapshot power-flow wrapper for the IEEE 123 feeder.

This module is the project's ground-truth physics simulator: it compiles the
vendored OpenDSS master file, solves a *snapshot* power flow (with regulator
tap-limiting matching the canonical ``Run_IEEE123Bus.DSS`` recipe so the
solution actually converges on Kersting's test case), and returns clean,
pickle-friendly Python data structures — per-bus voltages, per-line loadings,
total losses, and solver metadata.

Design
------
* **Stateless by contract.** Every call to :func:`run_snapshot` does a
  ``Clear`` + ``Redirect`` so previous overrides never leak. The OpenDSS
  engine itself is a global singleton, so the module is *not* safe to call
  from multiple threads concurrently; a :class:`threading.Lock` serialises
  access within a single process.
* **Bus-name style** matches :mod:`gridsense.topology`: phase suffixes are
  stripped (``150.1.2.3`` → ``150``) and names are lowercased. Per-phase
  voltages are averaged to a single scalar per bus.
* **Overrides** (``overrides={"42": 1.5}``) scale ``kW`` and ``kvar`` on
  every Load element whose ``Bus1`` resolves to the given bus. Scaling is
  applied *after* the initial Redirect and *before* Solve — no ``.dss``
  files are ever mutated.

Entry point
-----------
:func:`run_snapshot` returns a :class:`SnapshotResult` dataclass.

Example
-------
>>> from gridsense.power_flow import run_snapshot
>>> result = run_snapshot()
>>> result.converged
True
>>> sorted(result.bus_voltages_pu)[:3]
['1', '10', '100']
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import opendssdirect as dss

from gridsense.topology import DEFAULT_ROOT, MASTER_FILENAME

__all__ = ["SnapshotResult", "run_snapshot"]


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotResult:
    """Output of a single IEEE 123 snapshot power-flow solve.

    Attributes:
        bus_voltages_pu: ``bus_name -> per-unit voltage magnitude``, averaged
            across energised phases. Keys match
            :func:`gridsense.topology.load_ieee123` node keys (lowercase,
            phase suffix stripped). OpenDSS's virtual ``sourcebus`` is
            excluded because it is not present in the topology graph either.
        line_loadings_pct: ``line_element_name -> percent of normal amps``
            (0-100+). Computed as
            ``100 * max(terminal-1 phase currents) / NormAmps`` per Line.
            Elements with ``NormAmps == 0`` are omitted.
        total_losses_kw: Total real power losses in kW.
        total_losses_kvar: Total reactive power losses in kvar.
        converged: Whether ``Solution.Converged()`` returned True.
        iterations: Number of solver iterations used.
    """

    bus_voltages_pu: dict[str, float] = field(default_factory=dict)
    line_loadings_pct: dict[str, float] = field(default_factory=dict)
    total_losses_kw: float = 0.0
    total_losses_kvar: float = 0.0
    converged: bool = False
    iterations: int = 0


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

# The OpenDSS engine is a process-wide singleton. Serialise access so
# parallel test runners / dashboard callers don't clobber each other.
_ENGINE_LOCK = threading.Lock()

# Regulator-control recipe from ``data/ieee123/Run_IEEE123Bus.DSS`` — without
# it the solver runs out of control iterations on the canonical test case.
_REGCONTROL_COMMANDS: tuple[str, ...] = (
    "RegControl.creg1a.maxtapchange=1",
    "RegControl.creg2a.maxtapchange=1",
    "RegControl.creg3a.maxtapchange=1",
    "RegControl.creg4a.maxtapchange=1",
    "RegControl.creg3c.maxtapchange=1",
    "RegControl.creg4b.maxtapchange=1",
    "RegControl.creg4c.maxtapchange=1",
    "Set MaxControlIter=30",
)

# EnergyMeter is required by some reporting paths and harmless here.
_ENERGYMETER_COMMAND = "New EnergyMeter.Feeder Line.L115 1"


def _strip_bus_name(raw: str) -> str:
    """Mirror :func:`gridsense.topology._strip_bus_name`: ``150.1.2.3`` → ``150``."""
    return raw.split(".", 1)[0].strip().lower()


def _compile_circuit(master: Path) -> None:
    """Clear the engine and redirect into the master ``.dss`` file."""
    dss.Text.Command("Clear")
    # Quote the path so spaces (however unlikely) don't break the parser.
    dss.Text.Command(f'Redirect "{master}"')
    for cmd in _REGCONTROL_COMMANDS:
        dss.Text.Command(cmd)
    dss.Text.Command(_ENERGYMETER_COMMAND)


def _apply_overrides(overrides: Mapping[str, float]) -> None:
    """Scale kW/kvar on every Load whose Bus1 matches an override key.

    Mutates the in-memory OpenDSS model only — no files are touched. The
    next :func:`_compile_circuit` call resets everything.
    """
    if not overrides:
        return

    # Normalise keys to the same strip/lowercase convention as topology.
    wanted = {_strip_bus_name(bus): float(scale) for bus, scale in overrides.items()}

    # Walk every Load once, collect (name, scale) for matches, then apply.
    # We can't scale while iterating because :func:`dss.Loads.Name` setter
    # is how the active load is re-selected, and ``.Next()`` relies on the
    # current cursor position being untouched.
    scales: list[tuple[str, float]] = []
    idx = dss.Loads.First()
    while idx > 0:
        load_name = dss.Loads.Name()
        bus_raw = dss.CktElement.BusNames()[0] if dss.CktElement.BusNames() else ""
        bus = _strip_bus_name(bus_raw)
        if bus in wanted:
            scales.append((load_name, wanted[bus]))
        idx = dss.Loads.Next()

    for load_name, scale in scales:
        # Re-activate the load by name, then scale its real & reactive power.
        dss.Loads.Name(load_name)
        dss.Loads.kW(dss.Loads.kW() * scale)
        dss.Loads.kvar(dss.Loads.kvar() * scale)


def _collect_bus_voltages() -> dict[str, float]:
    """Average per-phase pu magnitudes to a single scalar per bus."""
    node_names = dss.Circuit.AllNodeNames()  # e.g. ['150.1', '150.2', '150.3', ...]
    mags = dss.Circuit.AllBusMagPu()
    if len(node_names) != len(mags):
        # Defensive — OpenDSSDirect guarantees alignment, but bail loudly if not.
        raise RuntimeError(
            f"AllNodeNames/AllBusMagPu length mismatch: {len(node_names)} vs {len(mags)}"
        )
    grouped: dict[str, list[float]] = defaultdict(list)
    for node, mag in zip(node_names, mags):
        bus = _strip_bus_name(node)
        grouped[bus].append(float(mag))
    return {bus: sum(v) / len(v) for bus, v in grouped.items() if v}


def _collect_line_loadings() -> dict[str, float]:
    """Compute percent-of-normal-amps for every Line element.

    Uses terminal-1 phase currents: ``CurrentsMagAng`` is a flat list of
    ``[mag, ang, mag, ang, ...]`` with ``num_conductors`` pairs per terminal.
    We take the maximum magnitude across terminal-1 conductors — this is
    the worst-case conductor loading, which is what matters for violation
    detection.
    """
    loadings: dict[str, float] = {}
    idx = dss.Lines.First()
    while idx > 0:
        name = dss.Lines.Name()
        norm_amps = float(dss.Lines.NormAmps() or 0.0)
        if norm_amps <= 0.0:
            idx = dss.Lines.Next()
            continue
        currents = dss.CktElement.CurrentsMagAng()
        num_cond = int(dss.CktElement.NumConductors() or 0)
        if num_cond <= 0 or not currents:
            idx = dss.Lines.Next()
            continue
        # Terminal 1 occupies the first 2*num_cond entries: [mag, ang] per conductor.
        term1_mags = [currents[2 * i] for i in range(num_cond) if 2 * i < len(currents)]
        max_mag = max(term1_mags) if term1_mags else 0.0
        loadings[name] = 100.0 * max_mag / norm_amps
        idx = dss.Lines.Next()
    return loadings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_snapshot(
    root: Path | None = None,
    overrides: Mapping[str, float] | None = None,
) -> SnapshotResult:
    """Solve a snapshot power flow on the IEEE 123 feeder and return results.

    The call is stateless: on entry the OpenDSS engine is cleared and the
    master ``.dss`` file is re-redirected, so any prior ``overrides`` from
    a previous invocation are fully discarded. Not safe for concurrent
    calls from multiple threads (serialised internally via a module-level
    :class:`threading.Lock`).

    Args:
        root: Directory containing ``IEEE123Master.dss`` and its includes.
            Defaults to :data:`gridsense.topology.DEFAULT_ROOT`.
        overrides: Optional ``{bus_name: scale}`` mapping. Every Load whose
            ``Bus1`` (with phase suffix stripped, lowercased) matches
            ``bus_name`` has its ``kW`` and ``kvar`` multiplied by ``scale``.
            Buses with no loads are silently ignored. Keys follow the same
            normalisation as :func:`gridsense.topology.load_ieee123`.

    Returns:
        A frozen :class:`SnapshotResult` with bus voltages, line loadings,
        losses, convergence flag, and iteration count.

    Raises:
        FileNotFoundError: If ``IEEE123Master.dss`` is missing under ``root``.
        RuntimeError: If OpenDSSDirect returns malformed data (node/voltage
            length mismatch, etc.).
    """
    resolved_root = (root or DEFAULT_ROOT).resolve()
    master = resolved_root / MASTER_FILENAME
    if not master.exists():
        raise FileNotFoundError(f"IEEE 123 master file not found: {master}")

    with _ENGINE_LOCK:
        _compile_circuit(master)
        _apply_overrides(overrides or {})
        dss.Solution.Solve()

        converged = bool(dss.Solution.Converged())
        iterations = int(dss.Solution.Iterations() or 0)

        # Circuit.Losses() returns [watts, vars] — convert to kW/kvar.
        raw_losses = dss.Circuit.Losses() or [0.0, 0.0]
        total_losses_kw = float(raw_losses[0]) / 1000.0
        total_losses_kvar = float(raw_losses[1]) / 1000.0

        bus_voltages = _collect_bus_voltages()
        line_loadings = _collect_line_loadings()

    return SnapshotResult(
        bus_voltages_pu=bus_voltages,
        line_loadings_pct=line_loadings,
        total_losses_kw=total_losses_kw,
        total_losses_kvar=total_losses_kvar,
        converged=converged,
        iterations=iterations,
    )
