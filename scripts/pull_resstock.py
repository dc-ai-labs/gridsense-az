#!/usr/bin/env python3
"""Pull NREL ResStock Maricopa County residential sample via DuckDB+httpfs.

PLAN.md §4.4. No auth. OEDI data lake is publicly-readable S3.

Implementation note (deviation from PLAN §4.4):
    The 2024 resstock_amy2018_release_2 schema partitions per-building timeseries
    by `by_state/upgrade=0/state=AZ/`, NOT by `by_county/state=AZ/county=G04013/`.
    County information lives on the `metadata/baseline.parquet` catalog as
    columns `in.county` (e.g. 'G0400130') and `in.county_name` ('Maricopa County').
    We therefore fetch a 500-row sample of Maricopa baseline metadata, which
    is the county-specific slice PLAN §4.4 targeted. Timeseries-per-building
    can be fetched in a second pass by bldg_id if/when feature engineering
    needs it (feature pipeline is a separate SDE task).

Writes data/raw/resstock/maricopa_sample.parquet with 500+ rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "resstock" / "maricopa_sample.parquet"

METADATA_S3 = (
    "s3://oedi-data-lake/nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/2024/"
    "resstock_amy2018_release_2/metadata/baseline.parquet"
)


def _build_sql(s3_path: str, out: Path, limit: int) -> str:
    # Double-quote columns that contain dots.
    return f"""
        INSTALL httpfs; LOAD httpfs;
        SET s3_region='us-west-2';
        COPY (
            SELECT *
            FROM read_parquet('{s3_path}')
            WHERE "in.state" = 'AZ'
              AND "in.county_name" = 'Maricopa County'
            LIMIT {limit}
        ) TO '{out}' (FORMAT PARQUET);
    """


def _build_explain_sql(s3_path: str, limit: int) -> str:
    return f"""
        EXPLAIN SELECT *
        FROM read_parquet('{s3_path}')
        WHERE "in.state" = 'AZ'
          AND "in.county_name" = 'Maricopa County'
        LIMIT {limit}
    """


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pull ResStock Maricopa County baseline sample.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--s3-path", default=METADATA_S3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--explain",
        action="store_true",
        help="Validate SQL via EXPLAIN without executing the COPY (used by tests).",
    )
    args = ap.parse_args(argv)

    import duckdb

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0 and not args.force and not args.explain:
        print(f"[pull_resstock] skip — {out} already exists ({out.stat().st_size:,} bytes)")
        return 0

    print(f"[pull_resstock] out={out} limit={args.limit}")
    print(f"[pull_resstock] s3_path={args.s3_path}")

    con = duckdb.connect()
    try:
        con.sql("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
        # httpfs timeouts — baseline.parquet is ~500 MB; be generous.
        try:
            con.sql("SET http_timeout=600000;")  # 10 min
        except Exception:
            pass  # older duckdb may not expose this setting
        try:
            con.sql("SET http_retries=5;")
        except Exception:
            pass
        if args.explain:
            plan = con.sql(_build_explain_sql(args.s3_path, args.limit)).fetchall()
            print("[pull_resstock] EXPLAIN ok — plan rows:", len(plan))
            return 0
        con.sql(_build_sql(args.s3_path, out, args.limit))
    except Exception as exc:  # noqa: BLE001
        print(f"[pull_resstock] ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        con.close()

    if not out.exists():
        print("[pull_resstock] ERROR: output parquet not written", file=sys.stderr)
        return 1

    # Sanity check — row count.
    con2 = duckdb.connect()
    try:
        (n,) = con2.sql(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()
    finally:
        con2.close()

    if n < args.limit:
        print(
            f"[pull_resstock] FAIL sanity check: {n} rows < {args.limit}",
            file=sys.stderr,
        )
        return 2
    print(f"[pull_resstock] ok rows={n:,} bytes={out.stat().st_size:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
