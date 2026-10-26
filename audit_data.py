#!/usr/bin/env python3
"""Summarize provider data quality *without changing* any raw datasets.

Requires pandas. Output: data/quality/file_quality.csv and player_coverage.csv
"""
from pathlib import Path
import argparse
import re

import pandas as pd

FINALISTS_2026 = [
    "Elena Rybakina", "Aryna Sabalenka", "Mirra Andreeva", "Jessica Pegula",
    "Coco Gauff", "Elina Svitolina", "Karolina Muchova", "Linda Noskova",
]


def fraction_valid(series):
    return float(series.notna().mean()) if len(series) else float("nan")


def audit_file(path):
    frame = pd.read_csv(path, low_memory=False, dtype={"tourney_id": "string", "score": "string"})
    if not {"winner_name", "loser_name", "score"}.issubset(frame.columns):
        return None, []
    n = len(frame)
    scores = frame["score"].fillna("").astype(str).str.upper()
    nonfinish = scores.str.contains(r"\b(?:RET|W/O|WALKOVER|DEF|ABN|ABD)\b", regex=True)
    # The tourney_date field usually marks the tournament week, not each match's day.
    dates = pd.to_datetime(frame.get("tourney_date", pd.Series(index=frame.index, dtype="object")),
                           format="%Y%m%d", errors="coerce")
    stats_cols = {"w_svpt", "l_svpt"}
    stats_present = pd.Series(False, index=frame.index)
    if stats_cols.issubset(frame.columns):
        serve_w = pd.to_numeric(frame["w_svpt"], errors="coerce")
        serve_l = pd.to_numeric(frame["l_svpt"], errors="coerce")
        stats_present = (serve_w > 0) & (serve_l > 0)
    rank_present = pd.Series(False, index=frame.index)
    if {"winner_rank", "loser_rank"}.issubset(frame.columns):
        rw = pd.to_numeric(frame["winner_rank"], errors="coerce")
        rl = pd.to_numeric(frame["loser_rank"], errors="coerce")
        rank_present = (rw > 0) & (rl > 0)
    keys = [x for x in ("tourney_id", "match_num", "winner_id", "loser_id") if x in frame]
    duplicates = int(frame.duplicated(subset=keys).sum()) if keys else 0
    row = {
        "source_file": str(path), "rows": n, "unique_winners": frame["winner_name"].nunique(),
        "year_min": dates.min().year if dates.notna().any() else "",
        "year_max": dates.max().year if dates.notna().any() else "",
        "unknown_tourney_date_pct": round((1 - fraction_valid(dates)) * 100, 2),
        "scores_missing_pct": round(scores.eq("").mean() * 100, 2),
        "noncompleted_marker_pct": round(nonfinish.mean() * 100, 2),
        "rankings_both_available_pct": round(rank_present.mean() * 100, 2),
        "both_serve_points_available_pct": round(stats_present.mean() * 100, 2),
        "duplicate_key_rows": duplicates,
        "hard_court_pct": round(frame.get("surface", pd.Series(index=frame.index, dtype="object")).astype(str).str.lower().eq("hard").mean() * 100, 2),
    }
    names = frame["winner_name"].fillna("").str.casefold().str.strip()
    losers = frame["loser_name"].fillna("").str.casefold().str.strip()
    players = []
    for player in FINALISTS_2026:
        mask = names.eq(player.casefold()) | losers.eq(player.casefold())
        players.append({"source_file": str(path), "player": player, "matches": int(mask.sum()),
                        "hard_matches": int((mask & frame.get("surface", pd.Series(index=frame.index, dtype="object")).astype(str).str.casefold().eq("hard")).sum())})
    return row, players


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/quality"))
    args = parser.parse_args()
    paths = sorted(args.raw_dir.glob("*/*.csv"))
    rows, players = [], []
    for path in paths:
        if "rankings" in path.name or "players" in path.name or "manifest" in path.name:
            continue
        try:
            result, p = audit_file(path)
            if result is not None:
                rows.append(result)
                players.extend(p)
                print("AUDIT", path, result["rows"], "rows")
        except Exception as exc:
            print("SKIPPING", path, "error:", exc)
    if not rows:
        raise SystemExit("No match CSVs found. Run download_data.py first.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "file_quality.csv", index=False)
    pd.DataFrame(players).to_csv(args.out_dir / "player_coverage.csv", index=False)
    print("Wrote audit reports to", args.out_dir)
    print("Caution: this is an initial structural audit, not proof of true historical completeness.")


if __name__ == "__main__":
    main()
