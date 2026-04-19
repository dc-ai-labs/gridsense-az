"""Generate schema-valid stub forecast JSONs for the Next.js dashboard (Slice A).

Overwritten by Slice B's real precompute run. Produces:
  web/public/data/forecasts/tomorrow_baseline.json
  web/public/data/forecasts/tomorrow_heat.json
  web/public/data/forecasts/tomorrow_ev.json
  web/public/data/forecasts/feeder_topology.json
  web/public/data/forecasts/model_metrics.json
  web/public/data/forecasts/generated_at.json

Uses the real IEEE 123 topology from data/ieee123/ so the map renders
a faithful feeder layout even before real inference is wired.
"""
from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gridsense.topology import load_ieee123  # type: ignore  # noqa: E402

OUT = REPO / "web" / "public" / "data" / "forecasts"
OUT.mkdir(parents=True, exist_ok=True)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_quantiles(base_hour: list[float], generated: datetime) -> list[dict]:
    rows = []
    for h in range(24):
        ts = generated + timedelta(hours=h)
        p50 = base_hour[h]
        p10 = p50 * 0.88
        p90 = p50 * 1.14
        rows.append(
            {
                "ts": _iso(ts),
                "hour": h,
                "p10_mw": round(p10, 2),
                "p50_mw": round(p50, 2),
                "p90_mw": round(p90, 2),
            }
        )
    return rows


def _diurnal(peak_mw: float, peak_hour: int, trough_frac: float = 0.48) -> list[float]:
    """Synthetic 24h MW profile with peak at peak_hour."""
    trough = peak_mw * trough_frac
    out = []
    for h in range(24):
        # cosine half-cycle centered on peak_hour, normalized to [trough, peak]
        dh = min((h - peak_hour) % 24, (peak_hour - h) % 24)
        # distance 0..12 → cos factor 1..-1
        x = dh / 12.0 * math.pi
        k = (math.cos(x) + 1) / 2  # 1 at peak, 0 at trough
        out.append(trough + (peak_mw - trough) * k)
    return out


def _topology() -> dict:
    g = load_ieee123()
    names = sorted(g.nodes())
    xs = [g.nodes[n].get("x", 0.0) for n in names]
    ys = [g.nodes[n].get("y", 0.0) for n in names]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # Fallback for degenerate (two buses with 0,0 coords) — nudge them.
    rng = random.Random(1337)
    span_x = max(xmax - xmin, 1.0)
    span_y = max(ymax - ymin, 1.0)
    nodes = []
    for n, x, y in zip(names, xs, ys):
        if x == 0 and y == 0:
            x = xmin + span_x * rng.random()
            y = ymin + span_y * rng.random()
        nodes.append(
            {
                "bus": n,
                "x_norm": round((x - xmin) / span_x, 5),
                # flip Y so small-Y is top (SVG convention)
                "y_norm": round(1.0 - (y - ymin) / span_y, 5),
            }
        )
    edges = []
    for u, v, data in g.edges(data=True):
        kind = data.get("element_type", "line")
        edges.append(
            {
                "from": u,
                "to": v,
                "kind": "xfmr" if kind == "transformer" else (
                    "switch" if kind == "switch" else "line"
                ),
            }
        )
    return {"n_nodes": len(nodes), "nodes": nodes, "edges": edges}


def _per_bus(names: list[str], scenario: str, rng: random.Random) -> dict:
    """Return per_bus metrics. Heat/EV get amplified peaks + higher risk."""
    per_bus = {}
    # scenario-specific scale
    if scenario == "baseline":
        peak_mu, peak_sd = 320.0, 140.0
        risk_mu, risk_sd = 0.28, 0.16
    elif scenario == "heat":
        peak_mu, peak_sd = 470.0, 180.0
        risk_mu, risk_sd = 0.52, 0.20
    else:  # ev
        peak_mu, peak_sd = 430.0, 200.0
        risk_mu, risk_sd = 0.46, 0.22

    for n in names:
        peak = max(40.0, rng.gauss(peak_mu, peak_sd))
        rating = peak * rng.uniform(1.05, 1.6)
        risk = max(0.02, min(0.98, rng.gauss(risk_mu, risk_sd)))
        per_bus[n] = {
            "bus": n,
            "risk_score": round(risk, 3),
            "peak_load_kw": round(peak, 1),
            "rating_kw": round(rating, 1),
        }
    return per_bus


def _leaderboard(per_bus: dict, n: int = 10) -> list[dict]:
    rows = sorted(per_bus.values(), key=lambda r: r["risk_score"], reverse=True)[:n]
    out = []
    for r in rows:
        out.append(
            {
                "id": f"F-{r['bus']}_PHX",
                "bus": r["bus"],
                "risk_score": r["risk_score"],
                "peak_mw": round(r["peak_load_kw"] / 1000.0, 2),
            }
        )
    return out


def _opendss(scenario: str, names: list[str], rng: random.Random) -> dict:
    devs = []
    pool = rng.sample(names, 8)
    for b in pool:
        devs.append(
            {
                "bus": b,
                "vdev_pu": round(rng.uniform(-0.08, 0.08), 4),
            }
        )
    devs.sort(key=lambda r: abs(r["vdev_pu"]), reverse=True)
    if scenario == "baseline":
        overloads = []
    elif scenario == "heat":
        overloads = [
            {"element": "Line.L17_PHX", "loading_pct": 118.4, "limit_mva": 3.8},
            {"element": "Line.L22_N", "loading_pct": 104.2, "limit_mva": 4.2},
            {"element": "Transformer.T_SUB_4", "loading_pct": 101.7, "limit_mva": 6.0},
        ]
    else:  # ev
        overloads = [
            {"element": "Line.L17_PHX", "loading_pct": 109.6, "limit_mva": 3.8},
            {"element": "Line.L08_RES", "loading_pct": 106.3, "limit_mva": 2.5},
        ]
    return {
        "converged": True,
        "scenario": scenario,
        "top_bus_deviations": devs[:5],
        "overloads": overloads,
    }


def _actions(scenario: str) -> list[dict]:
    if scenario == "baseline":
        return [
            {"label": "MONITOR FEEDER_17_PHX SECONDARY", "severity": "primary"},
            {"label": "PRE-STAGE CREW NORTH_VALLEY_DEPOT", "severity": "primary"},
            {"label": "ROUTINE VOLTAGE REG CHECK 15:00", "severity": "primary"},
        ]
    if scenario == "heat":
        return [
            {"label": "SHED 40 MW NON-ESSENTIAL @ 17:00", "severity": "error"},
            {"label": "DISPATCH RESERVE GENERATION A-7", "severity": "error"},
            {"label": "ACTIVATE DEMAND RESPONSE TIER-2", "severity": "secondary"},
            {"label": "RECLOSE SWITCH S22 → FEEDER_17", "severity": "secondary"},
            {"label": "NOTIFY ASU CAMPUS CHILLER FLEET", "severity": "tertiary"},
        ]
    return [
        {"label": "TOU RATE SIGNAL RESIDENTIAL@19:00", "severity": "secondary"},
        {"label": "THROTTLE L2 CHARGER FLEET 30%", "severity": "secondary"},
        {"label": "STAND UP MOBILE XFMR @ SUB_12", "severity": "tertiary"},
        {"label": "MONITOR FEEDER_08_RES FOR THERMAL", "severity": "primary"},
    ]


def _weather(scenario: str) -> dict:
    if scenario == "baseline":
        return {"peak_temp_f": 108.0, "peak_hour": 15, "source": "synthetic"}
    if scenario == "heat":
        return {"peak_temp_f": 118.0, "peak_hour": 15, "source": "synthetic"}
    return {"peak_temp_f": 108.0, "peak_hour": 15, "source": "synthetic"}


TOP_DRIVERS = [
    {"name": "temp_c", "ig": 0.42},
    {"name": "hour_sin", "ig": 0.21},
    {"name": "dewpoint_c", "ig": 0.15},
    {"name": "is_weekend", "ig": 0.08},
    {"name": "dow_sin", "ig": 0.07},
]


def _forecast(scenario: str, generated: datetime, topo: dict) -> dict:
    rng = random.Random({"baseline": 11, "heat": 22, "ev": 33}[scenario])
    names = [n["bus"] for n in topo["nodes"]]

    if scenario == "baseline":
        peak_mw, peak_hour, capacity = 3500.0, 17, 4600.0
    elif scenario == "heat":
        peak_mw, peak_hour, capacity = 4900.0, 17, 4600.0
    else:
        peak_mw, peak_hour, capacity = 4400.0, 19, 4600.0

    profile = _diurnal(peak_mw, peak_hour)
    quantiles = _build_quantiles(profile, generated)
    per_bus = _per_bus(names, scenario, rng)
    leaderboard = _leaderboard(per_bus)

    return {
        "scenario": scenario,
        "generated_at": _iso(generated),
        "quantiles": quantiles,
        "per_bus": per_bus,
        "risk_leaderboard": leaderboard,
        "feeder_rollup": {
            "peak_mw": round(peak_mw, 1),
            "peak_hour": peak_hour,
            "capacity_mw": round(capacity, 1),
            "load_factor": round(peak_mw / capacity, 3),
        },
        "opendss": _opendss(scenario, names, rng),
        "weather": _weather(scenario),
        "top_drivers": TOP_DRIVERS,
        "recommended_actions": _actions(scenario),
    }


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    # Use tomorrow 00:00 UTC as the aligned forecast start.
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    topo = _topology()

    (OUT / "feeder_topology.json").write_text(json.dumps(topo, indent=2))

    for scen in ("baseline", "heat", "ev"):
        doc = _forecast(scen, tomorrow, topo)
        (OUT / f"tomorrow_{scen}.json").write_text(json.dumps(doc, indent=2))

    metrics_src = json.loads((REPO / "data" / "models" / "metrics.json").read_text())
    metrics = {
        "train_mae_kw": round(metrics_src["train_mae"], 2),
        "val_mae_kw": round(metrics_src["val_mae"], 2),
        "test_mae_kw": round(metrics_src["test_mae"], 2),
        "persistence_mae_kw": round(metrics_src["baseline_mae"], 2),
        "improvement_pct": round(metrics_src["improvement_pct"], 2),
        "n_params": int(metrics_src["n_params"]),
        "top_drivers": TOP_DRIVERS,
    }
    (OUT / "model_metrics.json").write_text(json.dumps(metrics, indent=2))

    (OUT / "generated_at.json").write_text(
        json.dumps(
            {
                "iso": _iso(now),
                "nws_source": "synthetic",
                "hours_forecast": 24,
                "git_sha": _git_sha(),
            },
            indent=2,
        )
    )

    print(f"Wrote stubs to {OUT}")
    for p in sorted(OUT.glob("*.json")):
        print(f"  {p.name} ({p.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
