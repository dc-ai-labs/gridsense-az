#!/usr/bin/env python3
"""Pull NREL NSRDB PSM3 irradiance for Phoenix (lat 33.45, lon -112.07) 2020-2023.

PLAN.md §4.2. Requires NREL_API_KEY in env — gracefully SKIPs (exit 0) when missing
so pull_all.sh can proceed.

Writes one CSV per year to data/raw/nsrdb/phoenix_{YEAR}.csv.
Sanity check: file has >= 8500 rows (allowing leap-year + missing-row slack on the
8760-hour calendar).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "nsrdb"
DEFAULT_YEARS = [2020, 2021, 2022, 2023]
NSRDB_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"
DEFAULT_EMAIL = "dchanda1@asu.edu"
DEFAULT_WKT = "POINT(-112.07 33.45)"
DEFAULT_ATTRS = "ghi,dhi,dni,air_temperature,relative_humidity"
USER_AGENT = "GridSense-AZ/0.1 (dchanda1@asu.edu; APS AI-for-Energy hackathon)"
MIN_ROWS = 8500


def _fetch_year(
    year: int,
    out_dir: Path,
    api_key: str,
    email: str,
    force: bool,
    timeout: int = 60,
) -> Path:
    out = out_dir / f"phoenix_{year}.csv"
    if out.exists() and out.stat().st_size > 0 and not force:
        print(f"  [skip] {out.name} already exists ({out.stat().st_size:,} bytes)")
        return out

    params = {
        "wkt": DEFAULT_WKT,
        "names": str(year),
        "interval": "60",
        "attributes": DEFAULT_ATTRS,
        "api_key": api_key,
        "email": email,
        "utc": "false",
    }
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(
                NSRDB_URL,
                params=params,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT, "Accept": "text/csv"},
            )
            resp.raise_for_status()
            if len(resp.content) < 1024:
                raise RuntimeError(f"suspiciously small payload: {len(resp.content)} bytes")
            out.write_bytes(resp.content)
            print(f"  [ok]   {out.name} ({len(resp.content):,} bytes)")
            return out
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < 3:
                delay = 2**attempt
                print(f"  [retry {attempt}/3 in {delay}s] {year}: {exc}", file=sys.stderr)
                time.sleep(delay)
    raise RuntimeError(f"NSRDB fetch failed for {year} after 3 attempts: {last_err}")


def _row_count(csv_path: Path) -> int:
    """PSM3 CSVs have 2 header lines (metadata + columns), then data."""
    with csv_path.open("rb") as f:
        n = sum(1 for _ in f)
    return max(n - 2, 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pull NREL NSRDB PSM3 for Phoenix.")
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        help="Years to pull (default: 2020-2023).",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--email", default=DEFAULT_EMAIL, help="Email for NREL API.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    api_key = os.environ.get("NREL_API_KEY", "").strip()
    if not api_key:
        print("SKIP: set NREL_API_KEY in .env (NSRDB is a keyed NREL endpoint).")
        return 0

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pull_nsrdb] years={args.years} out={out_dir} email={args.email}")
    fetched: list[Path] = []
    for year in args.years:
        try:
            fetched.append(_fetch_year(year, out_dir, api_key, args.email, args.force))
        except Exception as exc:  # noqa: BLE001
            print(f"[pull_nsrdb] ERROR on year {year}: {exc}", file=sys.stderr)

    if not fetched:
        print("[pull_nsrdb] no years fetched", file=sys.stderr)
        return 1

    sample = fetched[0]
    rows = _row_count(sample)
    if rows < MIN_ROWS:
        print(
            f"[pull_nsrdb] FAIL sanity check on {sample.name}: {rows} rows < {MIN_ROWS}",
            file=sys.stderr,
        )
        return 2

    total = sum(_row_count(p) for p in fetched)
    print(f"[pull_nsrdb] ok N_years={len(fetched)} sample_rows={rows} total_rows=~{total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
