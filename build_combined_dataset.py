"""Build the canonical, de-duplicated WTA match dataset.

This script preserves raw provider files and produces one output row for every
normalized match identity.  It intentionally does not create any model,
features, labels, or train/test splits.

Run from the project directory:
    python build_combined_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from environmental_data_pipeline import enrich_matches_with_environment
from step_02_load_and_compatre_providers import load_provider


OUTPUT_PATH = Path("data/processed/wta_matches_combined_deduplicated.csv")
SOURCE_SPECS = {
    "Sackmann": ("sackmann", "wta_matches_*.csv"),
    "TennisMyLife": ("tennismylife", "*_wta.csv"),
}
IDENTITY_COLUMNS = [
    "tourney_date",
    "tourney_name",
    "round",
    "winner_name",
    "loser_name",
]
KEY_COLUMNS = [
    "tourney_date",
    "tourney_name",
    "winner_name",
    "loser_name",
    "score",
    "surface",
    "winner_rank",
    "loser_rank",
    "w_svpt",
    "l_svpt",
]
SERVE_STAT_COLUMNS = [
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
]
QUALITY_COLUMNS = list(dict.fromkeys(KEY_COLUMNS + SERVE_STAT_COLUMNS))
PROVIDER_TIE_BREAK = {"TennisMyLife": 1, "Sackmann": 0}
INCOMPLETE_SCORE_PATTERN = r"\b(?:RET|W/O|WALKOVER|DEF|ABN|ABD)\b"


def normalized_identity_part(series: pd.Series) -> pd.Series:
    """Normalize text consistently before cross-provider identity matching."""
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_source_rows() -> pd.DataFrame:
    """Load both raw providers and join only the cached weather enrichment."""
    frames = []
    for label, (provider, pattern) in SOURCE_SPECS.items():
        frame = load_provider(provider, pattern).copy()
        frame["provider_label"] = label
        frames.append(frame)
    return enrich_matches_with_environment(pd.concat(frames, ignore_index=True))


def prepare_source_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Add transparent identity, quality, and availability fields to source rows."""
    prepared = matches.copy()
    identity_parts = [
        prepared["tourney_date"].astype("string"),
        *[
            normalized_identity_part(prepared[column])
            for column in IDENTITY_COLUMNS[1:]
        ],
    ]
    prepared["match_identity"] = identity_parts[0]
    for part in identity_parts[1:]:
        prepared["match_identity"] += " | " + part

    score = prepared["score"].fillna("").astype(str).str.strip().str.upper()
    prepared["match_status"] = "Completed"
    prepared.loc[score.eq(""), "match_status"] = "Unknown"
    prepared.loc[
        score.str.contains(INCOMPLETE_SCORE_PATTERN, regex=True), "match_status"
    ] = "Incomplete"

    winner_rank = pd.to_numeric(prepared["winner_rank"], errors="coerce")
    loser_rank = pd.to_numeric(prepared["loser_rank"], errors="coerce")
    prepared["both_rankings_available"] = (winner_rank > 0) & (loser_rank > 0)

    winner_serve_points = pd.to_numeric(prepared["w_svpt"], errors="coerce")
    loser_serve_points = pd.to_numeric(prepared["l_svpt"], errors="coerce")
    prepared["both_serve_stats_available"] = (
        (winner_serve_points > 0) & (loser_serve_points > 0)
    )
    prepared["record_completeness"] = prepared[QUALITY_COLUMNS].notna().sum(axis=1)
    prepared["provider_tie_break"] = prepared["provider_label"].map(
        PROVIDER_TIE_BREAK
    )
    return prepared


def combine_source_rows(matches: pd.DataFrame) -> pd.DataFrame:
    """Retain one best-quality record for each match and preserve provenance."""
    ordered = matches.sort_values(
        ["match_identity", "record_completeness", "provider_tie_break"],
        ascending=[True, False, False],
        kind="stable",
    )
    provenance = (
        ordered.groupby("match_identity", as_index=False)
        .agg(
            source_record_count=("provider_label", "size"),
            source_providers=(
                "provider_label",
                lambda providers: ", ".join(sorted(set(providers))),
            ),
            source_files=(
                "source_file",
                lambda files: ", ".join(sorted(set(files))),
            ),
        )
    )
    combined = ordered.drop_duplicates("match_identity", keep="first").copy()
    combined = combined.merge(provenance, on="match_identity", how="left")
    combined = combined.rename(columns={"source_file": "retained_source_file"})
    combined["retained_provider"] = combined["provider_label"]
    combined["provider"] = "combined"
    combined["provider_label"] = "Combined"
    combined["is_cross_source_duplicate"] = combined["source_record_count"].gt(1)
    combined["deduplication_rule"] = (
        "normalized tournament-week, tournament, round, winner, loser; "
        "most complete record; TennisMyLife tie-break"
    )
    combined = combined.drop(columns=["provider_tie_break"])
    if combined["match_identity"].duplicated().any():
        raise ValueError("Combined output still contains duplicate match identities.")
    return combined.sort_values(
        ["source_year", "tourney_date", "tourney_name", "round", "winner_name", "loser_name"],
        kind="stable",
    ).reset_index(drop=True)


def build_combined_dataset(output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """Create the persisted canonical dataset from the unchanged raw inputs."""
    combined = combine_source_rows(prepare_source_rows(load_source_rows()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output CSV path (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()
    combined = build_combined_dataset(args.output)
    print(f"Wrote {len(combined):,} unique matches to {args.output}")
    print(
        "Cross-source duplicate identities consolidated: "
        f"{int(combined['is_cross_source_duplicate'].sum()):,}"
    )
    print("Retained provider counts:")
    print(combined["retained_provider"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
