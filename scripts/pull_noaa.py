#!/usr/bin/env python3
"""Pull NOAA ISD hourly weather for KPHX (Phoenix Sky Harbor) 2019-2025.

PLAN.md §4.1. Writes data/raw/noaa/KPHX_{year}.csv. No auth. Idempotent —
skip any year whose target file already exists unless --force is passed.

Station KPHX: USAF 722780, WBAN 23183 -> 72278023183.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "noaa"
STATION = "72278023183"
URL_TMPL = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station}.csv"
DEFAULT_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
USER_AGENT = "GridSense-AZ/0.1 (dchanda1@asu.edu; APS AI-for-Energy hackathon)"
MIN_ROWS_PER_YEAR = 8000


def _fetch_year(year: int, out_dir: Path, force: bool, timeout: int = 30) -> Path:
    """Fetch one year of KPHX ISD CSV with exponential backoff."""
    out = out_dir / f"KPHX_{year}.csv"
    if out.exists() and out.stat().st_size > 0 and not force:
        print(f"  [skip] {out.name} already exists ({out.stat().st_size:,} bytes)")
        return out

    url = URL_TMPL.format(year=year, station=STATION)
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT, "Accept": "text/csv"},
            )
            resp.raise_for_status()
            if not resp.content or len(resp.content) < 1024:
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
    raise RuntimeError(f"NOAA fetch failed for {year} after 3 attempts: {last_err}")


def _sanity_check_one(csv_path: Path) -> int:
    """Return row count of a CSV (excluding header)."""
    with csv_path.open("rb") as f:
        # Quick line count; header subtracted.
        n = sum(1 for _ in f) - 1
    return max(n, 0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pull NOAA ISD KPHX hourly weather.")
    ap.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        help="Years to pull (default: 2019-2025).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: data/raw/noaa/).",
    )
    ap.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pull_noaa] station=KPHX years={args.years} out={out_dir}")
    fetched: list[Path] = []
    for year in args.years:
        path = _fetch_year(year, out_dir, args.force)
        fetched.append(path)

    if not fetched:
        print("[pull_noaa] no years fetched", file=sys.stderr)
        return 1

    sample = fetched[0]
    rows = _sanity_check_one(sample)
    if rows < MIN_ROWS_PER_YEAR:
        print(
            f"[pull_noaa] FAIL sanity check on {sample.name}: {rows} rows < {MIN_ROWS_PER_YEAR}",
            file=sys.stderr,
        )
        return 2

    total_bytes = sum(p.stat().st_size for p in fetched)
    total_rows_est = sum(_sanity_check_one(p) for p in fetched)
    print(
        f"[pull_noaa] ok N_years={len(fetched)} total_rows=~{total_rows_est} "
        f"total_bytes={total_bytes:,}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
