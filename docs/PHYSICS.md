# Physics — why GridSense-AZ is grounded, not just a regression

A machine-learned forecast that doesn't respect grid physics is a liability
on an operator's desk. GridSense-AZ closes the loop: every ML forecast is
run through a power-flow solver on a real test feeder before it is surfaced.
This document explains the physics we rely on, how we enforce it, and why
that matters for deployment.

## Power flow — first principles

Steady-state power flow is governed by three classical laws applied to a
network of nodes (buses) and edges (lines):

### Kirchhoff's Current Law (KCL)
At every bus, net current in = net current out. In power-flow terms, the
complex power injected by generation equals the complex power consumed by
loads plus losses:

```
  Σ S_gen(bus) − Σ S_load(bus) − Σ S_flow(bus) = 0
```

### Kirchhoff's Voltage Law (KVL)
Around any closed loop, the sum of voltage drops is zero.

### Ohm's Law (generalised)
For a line with complex impedance `Z = R + jX`, the current phasor is:

```
  I = (V_from − V_to) / Z
```

And the apparent power through the line:

```
  S_flow = V · I*    (complex conjugate current)
```

### Newton-Raphson for distribution feeders
Because loads and generators mix PQ (known power) and PV (known power +
voltage) bus types, the resulting system of non-linear equations is solved
iteratively. **OpenDSS** — the EPRI-authored distribution-system simulator
we use — runs a current-injection Newton-Raphson variant that handles
unbalanced three-phase networks natively, which is essential for real
distribution feeders (primary laterals are routinely single- or two-phase).

## IEEE 123-bus test feeder

The IEEE 123 Node Test Feeder is the de-facto research benchmark published by
the IEEE PES Distribution System Analysis Subcommittee (DSASC). It is used in
hundreds of papers because it captures, at manageable scale, the pathology
of a real radial distribution network:

- **4.16 kV nominal medium voltage.**
- **Unbalanced three-phase** mainline plus single- and two-phase laterals.
- **Voltage regulators** (auto-taps) at four locations.
- **Shunt capacitors** for power-factor correction.
- **Switches** (normally-closed and normally-open tie) for reconfiguration
  study.
- **85 spot loads and 40+ distributed loads** totalling ~3.5 MW nominal.

Citation: IEEE PES DSASC, "IEEE 123 Node Test Feeder," available via the
[resources page](https://cmte.ieee.org/pes-testfeeders/resources/). Files
shipped under `data/ieee123/` include the master DSS and BusCoords.

Why this feeder? It is standard, public, and well-characterised, so a reviewer
can reproduce our OpenDSS output exactly. Our ingested topology exposes
**132 buses** once regulator / switch internal nodes are included — matches
the `n_nodes: 132` in `data/models/metrics.json`.

## How GridSense-AZ uses OpenDSS

Code: [`src/gridsense/power_flow.py`](../src/gridsense/power_flow.py), function
`run_snapshot(root, overrides)`.

```
           per-bus load multipliers
                     │
                     ▼
           OpenDSSDirect.py binding
                     │
                     ▼
     solve(mode = snap, algorithm = Newton)
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
    bus_voltages_pu      line_loadings_pct
       (dict[str→float])   (dict[elem→%])
                     │
                     ▼
           total_losses_kw, converged flag
                     │
                     ▼
              SnapshotResult
```

One snapshot is run **per scenario** (baseline / heat / EV) at that
scenario's **peak hour**. This is the moment of maximum stress — if the
network survives the peak, it survives the other 23 hours. Pre-compute runs
solve the three scenarios in parallel.

Each `per-bus load multiplier` is computed as:

```
multiplier = peak_load_kw (forecast) / nominal_kw (topology)
```

so the model's forecast directly drives the physics solve — no hand-tuned
scaling.

## Voltage limits — ANSI C84.1

Per ANSI C84.1 "Electric Power Systems and Equipment — Voltage Ratings":

| Range | Utilisation voltage |
|---|---|
| Service Range A (normal) | 0.95 – 1.05 p.u. (± 5%) |
| Service Range B (occasional) | 0.917 – 1.058 p.u. |

We enforce Range A — any bus outside `[0.95, 1.05]` p.u. scores non-zero on
the voltage-deviation component of `risk_score` (see `docs/METRICS.md`). Above
ANSI's ±5% envelope, motors stall and electronics brown-out; below 0.95 p.u.
commercial HVAC compressors start seeing thermal protection trips.

## Thermal limits

Conductor thermal rating (`Normamps` in OpenDSS) is the maximum continuous
current before insulation ages prematurely. Our `line_loadings_pct` is
computed as:

```
  loading_pct = |I_line| / Normamps · 100
```

We flag anything over **100%** in the physics-check panel. In real operations
utilities will run a conductor 100–120% briefly; above that you lose anneal
resistance on ACSR strands or melt XLPE jacketing on cables. Our `risk_score`
factors a normalised loading at threshold 150% (the absolute worst-case
before a protective trip). Conservative defaults — operators can retune.

## Weather → load coupling

Residential and commercial AC load in Phoenix is roughly **quadratic in
ambient dry-bulb temperature above ~70 °F (21 °C)**:

```
P_cool ≈ k · max(T − T_ref, 0)²
```

This arises from the combination of (a) linear sensible-heat gain through
walls and (b) a compressor running closer to rated duty at higher outside
temperature (COP falls roughly linearly with condenser ambient). Our neural
net is not told this formula; it **learns it implicitly** from the exogenous
`temp_c` feature plus the per-bus historical load. The heat scenario's
+5.56 °C shift in `temp_c` plus a 12:00–22:00 cooling profile reproduces a
peak **≥ 1.3 × baseline** — asserted by `web/lib/validate.ts`.

## Why a physics check matters for deployment

An ML forecast alone can produce values that look statistically plausible but
violate electrical law — for example:

- A bus whose forecast drops to zero because the model extrapolates a holiday
  pattern, without registering that the downstream switch is closed and
  customers are still being served.
- A weekend-night forecast that sums to less than the plant's minimum-generation
  floor, which would crash the operator's unit-commitment optimiser.
- A heat-wave forecast that predicts a peak the conductors **cannot physically
  carry** without the operator being warned.

Running every scenario through OpenDSS catches the third class immediately.
The first two are still possible but the risk-score layer exposes them — a
bus where the model's p50 is low but the physics solve shows voltage out of
band is visibly anomalous in the UI.

Without a physics layer, an ML dispatch advisor is a regression tool. With
one, it becomes an operations tool.

## What's *not* modelled (honest)

- **Dynamic line rating (DLR)** — conductors sag and heat differently with
  wind and ambient; we use static `Normamps` only. FERC Order 881 requires
  ambient-adjusted ratings by 2025; a production build would add this.
- **Protection coordination** — relay curves, fuse saving, reclosers. Out of
  scope for a load-forecast product.
- **Harmonics and inverter-based resources** — IEEE 123 has no PV inverters;
  modern APS feeders do. Next iteration would add a PV penetration knob.
- **Unbalanced-load reallocation across phases** — we use the reference
  IEEE 123 phase allocation verbatim.
