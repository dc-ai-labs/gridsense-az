#!/usr/bin/env python3
"""Pull NREL EVI-Pro Lite daily load profiles — Phoenix climate parameter sweep.

PLAN.md §4.5. Requires NREL_API_KEY (graceful skip if missing).

Sweep: fleet_size in {100,500,1000,2000} x res_charging in {night, afternoon} = 8 scenarios.
Writes data/raw/evi_pro/phx_{FLEET}ev_{STRAT}.json per combo.

Sanity: all 8 files present and each parses as JSON with a 'load_profile' key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "evi_pro"
DEFAULT_FLEETS = [100, 500, 1000, 2000]
DEFAULT_STRATS = ["night", "afternoon"]
URL = "https://developer.nrel.gov/api/evi-pro-lite/v1/daily-load-profile"
USER_AGENT = "GridSense-AZ/0.1 (dchanda1@asu.edu; APS AI-for-Energy hackathon)"


def _fetch(fleet: int, strat: str, api_key: str, timeout: int = 60) -> dict:
    params = {
        "api_key": api_key,
        "fleet_size": fleet,
        "climate_zone": "very-hot-dry",
        "home_access_dist": "REAL_ESTATE",
        "home_power_dist": "MOSTLY_L2",
        "work_power_dist": "MOSTLY_L2",
        "res_charging": strat,
        "work_charging": "min_delay",
    }
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(
                URL,
                params=params,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < 3:
                delay = 2**attempt
                print(
                    f"  [retry {attempt}/3 in {delay}s] fleet={fleet} strat={strat}: {exc}",
                    file=sys.stderr,
                )
                time.sleep(delay)
    raise RuntimeError(
        f"EVI-Pro fetch failed for fleet={fleet} strat={strat} after 3 attempts: {last_err}"
    )


def _validate(payload: dict) -> bool:
    """A valid EVI-Pro Lite response carries a load profile (several synonyms possible)."""
    if not isinstance(payload, dict):
        return False
    # Top-level or nested under "result".
    keys = set(payload.keys())
    if "load_profile" in keys:
        return True
    res = payload.get("result")
    if isinstance(res, dict) and "load_profile" in res:
        return True
    # Also accept anything that contains per-hour arrays — EVI-Pro Lite sometimes
    # returns {"weekday_load_profile": [...], "weekend_load_profile": [...], ...}.
    if any("load_profile" in k for k in keys):
        return True
    if isinstance(res, dict) and any("load_profile" in k for k in res.keys()):
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pull NREL EVI-Pro Lite Phoenix sweep.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--fleets", nargs="+", type=int, default=DEFAULT_FLEETS)
    ap.add_argument("--strategies", nargs="+", default=DEFAULT_STRATS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    api_key = os.environ.get("NREL_API_KEY", "").strip()
    if not api_key:
        print("SKIP: set NREL_API_KEY in .env (EVI-Pro Lite is a keyed NREL endpoint).")
        return 0

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.fleets) * len(args.strategies)
    print(f"[pull_evi_pro] sweep fleets={args.fleets} strategies={args.strategies} total={total}")

    written: list[Path] = []
    for fleet in args.fleets:
        for strat in args.strategies:
            out = out_dir / f"phx_{fleet}ev_{strat}.json"
            if out.exists() and out.stat().st_size > 0 and not args.force:
                print(f"  [skip] {out.name} already exists ({out.stat().st_size:,} bytes)")
                written.append(out)
                continue
            try:
                payload = _fetch(fleet, strat, api_key)
            except Exception as exc:  # noqa: BLE001
                print(f"[pull_evi_pro] ERROR fleet={fleet} strat={strat}: {exc}", file=sys.stderr)
                continue
            out.write_text(json.dumps(payload, indent=2))
            size = out.stat().st_size
            ok = _validate(payload)
            marker = "ok" if ok else "WARN"
            print(f"  [{marker}]   {out.name} ({size:,} bytes, valid={ok})")
            written.append(out)

    if len(written) < total:
        print(
            f"[pull_evi_pro] FAIL — expected {total} files, got {len(written)}",
            file=sys.stderr,
        )
        return 2

    # Sanity: every file parses + validates.
    bad = []
    for p in written:
        try:
            data = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{p.name}: json error {exc}")
            continue
        if not _validate(data):
            bad.append(f"{p.name}: no 'load_profile' key")
    if bad:
        print("[pull_evi_pro] FAIL validation:", file=sys.stderr)
        for b in bad:
            print("  -", b, file=sys.stderr)
        return 3

    print(f"[pull_evi_pro] ok files={len(written)} out={out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
