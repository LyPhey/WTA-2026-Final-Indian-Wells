#!/usr/bin/env python3
"""Download WTA match data, keeping provider copies separate and unmodified.

This program is supplied as a reproducible *download utility*: the data belongs
 to its respective providers and their terms and attribution apply.

Example:
  python download_data.py --start-year 2016 --end-year 2026 --rankings
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

import requests

BASE_TML = "https://stats.tennismylife.org/data"
BASE_SACK = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main/wta"


def targets(root: Path, start: int, end: int, provider: str, rankings: bool):
    if provider in ("all", "tennismylife"):
        for year in range(start, end + 1):
            filename = f"{year}_wta.csv"
            yield "tennismylife", root / "tennismylife" / filename, f"{BASE_TML}/{filename}"
        filename = "wta_ongoing_tourneys.csv"
        yield "tennismylife", root / "tennismylife" / filename, f"{BASE_TML}/{filename}"
    if provider in ("all", "sackmann"):
        for year in range(start, end + 1):
            filename = f"wta_matches_{year}.csv"
            yield "sackmann", root / "sackmann" / filename, f"{BASE_SACK}/{filename}"
        yield "sackmann", root / "sackmann" / "wta_players.csv", f"{BASE_SACK}/wta_players.csv"
        if rankings:
            for decade in ("10s", "20s"):
                filename = f"wta_rankings_{decade}.csv"
                yield "sackmann", root / "sackmann" / filename, f"{BASE_SACK}/{filename}"


def is_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        cols = set(header)
        filename = path.name.removesuffix(".part")  # streamed download is saved as .csv.part
        if filename == "wta_players.csv":
            return {"player_id", "first_name", "last_name"}.issubset(cols)
        if filename.startswith("wta_rankings_"):
            return {"ranking_date", "ranking", "player_id"}.issubset(cols)
        return {"winner_name", "loser_name", "score", "tourney_date"}.issubset(cols)
    except (OSError, UnicodeError, csv.Error):
        return False


def fetch(session: requests.Session, url: str, path: Path) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    for retry in range(3):
        try:
            with session.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with temp.open("wb") as f:
                    for part in response.iter_content(chunk_size=256 * 1024):
                        if part:
                            f.write(part)
            if not is_csv(temp):
                raise ValueError("response does not look like the expected CSV format")
            temp.replace(path)
            return "downloaded", ""
        except (requests.RequestException, OSError, ValueError) as exc:
            temp.unlink(missing_ok=True)
            if retry == 2:
                return "error", str(exc)
            time.sleep(retry + 1)
    return "error", "unexpected downloader exit"


def sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--provider", choices=["all", "tennismylife", "sackmann"], default="all")
    parser.add_argument("--rankings", action="store_true", help="Download optional, potentially large, historical rankings files")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files (useful to update 2026 data)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without fetching")
    parser.add_argument("--out", type=Path, default=Path("data") / "raw")
    args = parser.parse_args()
    if not (1990 <= args.start_year <= args.end_year <= 2026):
        parser.error("choose a range between 1990 and 2026, with start <= end")
    plans = list(targets(args.out, args.start_year, args.end_year, args.provider, args.rankings))
    if args.dry_run:
        for provider, path, url in plans:
            print(f"{provider:13s} {path} <- {url}")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "WTA-ML-research-dataset-downloader/1.0 (noncommercial research)"})
    log = []
    for provider, path, url in plans:
        if path.exists() and not args.overwrite and is_csv(path):
            status, error = "cached", ""
        else:
            status, error = fetch(session, url, path)
        ok = path.exists() and is_csv(path)
        rec = {
            "utc_checked": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provider": provider,
            "source_url": url,
            "local_path": str(path),
            "status": status,
            "bytes": path.stat().st_size if ok else "",
            "sha256": sha256(path) if ok else "",
            "error": error,
        }
        log.append(rec)
        print(f"[{status:10}] {path.name}" + (f" - {error}" if error else ""))
    path = args.out / "download_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        writer.writeheader()
        writer.writerows(log)
    failures = sum(x["status"] == "error" for x in log)
    print(f"\nWrote {path} ; {len(log) - failures}/{len(log)} file(s) available")
    if failures:
        print("Some downloads failed. See manifest and try manually using the source links.", file=sys.stderr)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
