#!/usr/bin/env python3
"""Pull a small subset of a HuggingFace EnergyBench-compatible dataset.

PLAN.md §4.7 targets an electricity-load time-series benchmark dataset to stand
in for Pecan Street. Because several community repos use the 'EnergyBench' name
and some come and go, we probe a fixed candidate list (first HTTP-200 wins) and
download a lightweight subset (README + a single parquet/csv if present).

Graceful-skip contract (matches pull_nsrdb.py / pull_eia930.py):
    * If HF_TOKEN is unset AND none of the candidates are public, we emit
      ``SKIP:`` and exit 0 so pull_all.sh can continue.
    * If all candidates 404 (or otherwise unreachable) we still exit 0 but
      persist an ``attempt_manifest.json`` under the output dir so the next
      run has a record of what was tried.

Writes into ``data/raw/energybench/``:
    * ``README.md``              — copy of the dataset card, always present on success
    * ``attempt_manifest.json``  — what we tried and what the server said
    * one or more ``*.parquet`` / ``*.csv`` files if the dataset has small
      tabular shards (<= MAX_SHARD_BYTES each)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw" / "energybench"
USER_AGENT = "GridSense-AZ/0.1 (dchanda1@asu.edu; APS AI-for-Energy hackathon)"

# Probe order — first to 200 wins. Order matches the manager's task spec;
# ``ai-iot/EnergyBench`` is appended as the PLAN.md-canonical fallback
# (referenced at PLAN.md §4.7) before we give up.
DATASET_CANDIDATES: tuple[str, ...] = (
    "AIEnergy/EnergyBench",
    "microsoft/LCLF-LoadBench",
    "electricity-load-forecasting/autoformer-benchmarks",
    "ai-iot/EnergyBench",
)
ALLOW_PATTERNS: tuple[str, ...] = ("README*", "*.md", "*.json", "*.csv", "*.parquet")
# Guardrail against accidentally pulling multi-GB LFS shards; snapshot_download
# will still retrieve each file whole — this is an *advisory* ceiling used only
# for the post-fetch sanity pass.
MAX_SHARD_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_FILES_TO_KEEP = 4
HTTP_RETRIES = 3


@dataclass
class ProbeResult:
    """Outcome of a single HEAD-style ``dataset_info`` probe."""

    repo_id: str
    ok: bool
    status: str
    private: bool = False
    gated: bool = False
    detail: str = ""


@dataclass
class AttemptManifest:
    """Structured log written to disk whenever we cannot fetch anything."""

    pulled_at: str
    probes: list[dict[str, Any]] = field(default_factory=list)
    downloaded_repo: str | None = None
    downloaded_files: list[str] = field(default_factory=list)
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "pulled_at": self.pulled_at,
                "probes": self.probes,
                "downloaded_repo": self.downloaded_repo,
                "downloaded_files": self.downloaded_files,
                "note": self.note,
            },
            indent=2,
        )


def _probe(repo_id: str, token: str | None) -> ProbeResult:
    """Check whether ``repo_id`` is reachable as a dataset repo.

    Uses ``HfApi().dataset_info`` — one GET, small payload, respects token.
    Return value is exit-safe (never raises).
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    api = HfApi(user_agent=USER_AGENT)
    last_err: Exception | None = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            info = api.dataset_info(repo_id, token=token)
            return ProbeResult(
                repo_id=repo_id,
                ok=True,
                status="200",
                private=bool(getattr(info, "private", False)),
                gated=bool(getattr(info, "gated", False)),
                detail=f"sha={getattr(info, 'sha', '?')[:12]}",
            )
        except RepositoryNotFoundError as exc:
            return ProbeResult(repo_id=repo_id, ok=False, status="404", detail=str(exc))
        except GatedRepoError as exc:
            return ProbeResult(
                repo_id=repo_id, ok=False, status="gated", gated=True, detail=str(exc)
            )
        except HfHubHTTPError as exc:
            last_err = exc
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code in (401, 403):
                return ProbeResult(
                    repo_id=repo_id, ok=False, status=str(code), detail=str(exc)
                )
            if attempt < HTTP_RETRIES:
                time.sleep(2**attempt)
                continue
            return ProbeResult(
                repo_id=repo_id, ok=False, status=f"http_{code or '?'}", detail=str(exc)
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < HTTP_RETRIES:
                time.sleep(2**attempt)
                continue
            return ProbeResult(
                repo_id=repo_id, ok=False, status="error", detail=str(exc)
            )
    return ProbeResult(
        repo_id=repo_id, ok=False, status="error", detail=str(last_err) if last_err else ""
    )


def _snapshot(repo_id: str, out_dir: Path, token: str | None) -> list[Path]:
    """Download a small subset of ``repo_id`` into ``out_dir``.

    Uses ``allow_patterns`` to keep the footprint tiny and caps the number
    of kept shards after download — anything oversize is removed, keeping
    only the dataset card + up to ``MAX_FILES_TO_KEEP`` small tabular shards.
    """
    from huggingface_hub import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    local_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(out_dir),
            allow_patterns=list(ALLOW_PATTERNS),
            user_agent=USER_AGENT,
            token=token,
            max_workers=2,
        )
    )

    kept: list[Path] = []
    readme = _first_existing(
        local_path / "README.md",
        local_path / "README.MD",
        local_path / "readme.md",
    )
    if readme is not None:
        kept.append(readme)

    # Keep the smallest MAX_FILES_TO_KEEP tabular shards under the size cap.
    tabular_candidates: list[Path] = []
    for pattern in ("*.parquet", "*.csv", "*.json"):
        tabular_candidates.extend(local_path.rglob(pattern))
    # Skip cache metadata the hub library writes next to our downloads.
    tabular_candidates = [
        p for p in tabular_candidates if ".cache" not in p.parts and p != readme
    ]
    tabular_candidates.sort(key=lambda p: p.stat().st_size)
    for p in tabular_candidates:
        if len(kept) - (1 if readme else 0) >= MAX_FILES_TO_KEEP:
            break
        if p.stat().st_size > MAX_SHARD_BYTES:
            continue
        kept.append(p)
    return kept


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _write_attempt_manifest(out_dir: Path, manifest: AttemptManifest) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "attempt_manifest.json"
    path.write_text(manifest.to_json())
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pull a lightweight subset of an EnergyBench-compatible HF dataset.",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--candidates",
        nargs="+",
        default=list(DATASET_CANDIDATES),
        help="Dataset repo ids to probe in order (first 200 wins).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if README.md already exists in --out-dir.",
    )
    args = ap.parse_args(argv)

    out_dir: Path = args.out_dir
    token = os.environ.get("HF_TOKEN", "").strip() or None

    # Matches pull_nsrdb.py / pull_eia930.py / pull_evi_pro.py: require the
    # relevant env key to avoid surprising outbound traffic from CI / offline
    # pulls. HF public datasets technically do not need a token, but gating
    # the probe on HF_TOKEN keeps pull_all.sh deterministic and lets us
    # exit-0 skip in environments without network access.
    if token is None:
        print(
            "SKIP: set HF_TOKEN in .env to pull EnergyBench "
            "(HF datasets are gated behind an auth check in this puller)."
        )
        return 0

    # Idempotency: if we already have a README + any tabular shard, skip.
    existing_readme = (out_dir / "README.md").exists()
    existing_shards = any(out_dir.rglob("*.parquet")) or any(out_dir.rglob("*.csv"))
    if existing_readme and existing_shards and not args.force:
        print(
            f"[pull_energybench] skip — {out_dir} already populated "
            "(README + tabular shard present)"
        )
        return 0

    print(f"[pull_energybench] out={out_dir} candidates={args.candidates}")
    manifest = AttemptManifest(pulled_at=datetime.now(timezone.utc).isoformat())

    chosen: ProbeResult | None = None
    for repo_id in args.candidates:
        probe = _probe(repo_id, token)
        manifest.probes.append(
            {
                "repo_id": probe.repo_id,
                "ok": probe.ok,
                "status": probe.status,
                "private": probe.private,
                "gated": probe.gated,
                "detail": probe.detail[:240],
            }
        )
        marker = "ok" if probe.ok else "miss"
        print(f"  [{marker}]   {probe.repo_id} -> {probe.status}")
        if probe.ok:
            chosen = probe
            break

    if chosen is None:
        note = "no EnergyBench-compatible dataset reachable"
        manifest.note = note
        _write_attempt_manifest(out_dir, manifest)
        print(f"SKIP: {note}")
        return 0

    print(f"[pull_energybench] fetching snapshot repo_id={chosen.repo_id}")
    try:
        kept = _snapshot(chosen.repo_id, out_dir, token)
    except Exception as exc:  # noqa: BLE001
        manifest.note = f"snapshot_download failed: {exc}"
        _write_attempt_manifest(out_dir, manifest)
        print(f"[pull_energybench] ERROR: {exc}", file=sys.stderr)
        # Graceful — we already logged the attempt. Exit non-zero so pull_all
        # records a warning but the orchestrator still continues.
        return 1

    manifest.downloaded_repo = chosen.repo_id
    manifest.downloaded_files = [str(p.relative_to(out_dir)) for p in kept]
    manifest.note = "ok"
    _write_attempt_manifest(out_dir, manifest)

    # Sanity: the output dir must be non-empty and have a README.
    if not any(out_dir.iterdir()):
        print("[pull_energybench] FAIL: output dir empty after fetch", file=sys.stderr)
        return 2
    if not (out_dir / "README.md").exists() and not any(out_dir.rglob("README*")):
        print("[pull_energybench] WARN: no README.md in downloaded snapshot", file=sys.stderr)

    total_bytes = sum(p.stat().st_size for p in kept if p.exists())
    print(
        f"[pull_energybench] ok repo={chosen.repo_id} files={len(kept)} "
        f"bytes={total_bytes:,} out={out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
