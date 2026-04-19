#!/usr/bin/env python3
"""Pull EIA-930 balancing-authority hourly demand for AZPS 2019-present.

PLAN.md §4.3. Requires EIA_API_KEY in env; gracefully SKIPs (exit 0) when missing.

Paginates the v2 rto/region-data endpoint at 5000 rows/call. Writes one parquet
to data/raw/eia930/azps_demand.parquet. Sanity check: >= 50000 rows spanning
2019-present (i.e. >= 5-6 years of hourly data).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "eia930" / "azps_demand.parquet"
EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
USER_AGENT = "GridSense-AZ/0.1 (dchanda1@asu.edu; APS AI-for-Energy hackathon)"
PAGE_SIZE = 5000
MIN_ROWS = 50_000


def _page(url: str, params: dict, timeout: int = 60) -> list[dict]:
    """Fetch one page; retry with exponential backoff on HTTP errors."""
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            resp.raise_for_status()
            payload = resp.json()
            if "response" not in payload:
                raise RuntimeError(f"unexpected payload shape: {list(payload)[:5]}")
            return payload["response"].get("data", []) or []
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < 3:
                delay = 2**attempt
                print(f"  [retry {attempt}/3 in {delay}s] {exc}", file=sys.stderr)
                time.sleep(delay)
    raise RuntimeError(f"EIA fetch failed after 3 attempts: {last_err}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pull EIA-930 AZPS hourly demand.")
    ap.add_argument("--start", default="2019-01-01T00")
    ap.add_argument(
        "--end",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H"),
        help="End datetime in EIA format YYYY-MM-DDTHH (default: now).",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--respondent", default="AZPS")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    api_key = os.environ.get("EIA_API_KEY", "").strip()
    if not api_key:
        print("SKIP: set EIA_API_KEY in .env (EIA v2 API requires a free key).")
        return 0

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0 and not args.force:
        print(f"[pull_eia930] skip — {out} already exists ({out.stat().st_size:,} bytes)")
        return 0

    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": args.respondent,
        "facets[type][]": "D",  # D = Demand
        "start": args.start,
        "end": args.end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": PAGE_SIZE,
    }

    print(f"[pull_eia930] respondent={args.respondent} start={args.start} end={args.end}")
    rows: list[dict] = []
    offset = 0
    while True:
        page = _page(EIA_URL, {**params, "offset": offset})
        if not page:
            break
        rows.extend(page)
        print(f"  [page] offset={offset} fetched={len(page)} total={len(rows)}")
        offset += PAGE_SIZE
        if len(page) < PAGE_SIZE:
            break

    if not rows:
        print("[pull_eia930] ERROR: no rows returned", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)

    # Sanity — row count + span.
    n = len(df)
    period = pd.to_datetime(df["period"], errors="coerce")
    span = (period.min(), period.max())
    if n < MIN_ROWS:
        print(
            f"[pull_eia930] FAIL sanity check: {n} rows < {MIN_ROWS}",
            file=sys.stderr,
        )
        return 2
    print(f"[pull_eia930] ok rows={n:,} span={span[0]}..{span[1]} bytes={out.stat().st_size:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
