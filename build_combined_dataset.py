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
ALIAS_REPORT_PATH = Path("data/quality/tournament_alias_deduplication_report.csv")
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
ALIAS_CANDIDATE_COLUMNS = [
    "tourney_date",
    "round_normalized",
    "winner_name_normalized",
    "loser_name_normalized",
]
ALIAS_EVIDENCE_COLUMNS = ["score_normalized", "surface_normalized"]


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
    prepared["tourney_name_normalized"] = normalized_identity_part(
        prepared["tourney_name"]
    )
    prepared["round_normalized"] = normalized_identity_part(prepared["round"])
    prepared["winner_name_normalized"] = normalized_identity_part(
        prepared["winner_name"]
    )
    prepared["loser_name_normalized"] = normalized_identity_part(
        prepared["loser_name"]
    )
    prepared["score_normalized"] = normalized_identity_part(prepared["score"])
    prepared["surface_normalized"] = normalized_identity_part(prepared["surface"])
    identity_parts = [
        prepared["tourney_date"].astype("string"),
        prepared["tourney_name_normalized"],
        prepared["round_normalized"],
        prepared["winner_name_normalized"],
        prepared["loser_name_normalized"],
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


def joined_unique(values: pd.Series) -> str:
    """Return stable, human-readable provenance values for an audit row."""
    return ", ".join(sorted({str(value) for value in values.dropna() if str(value)}))


def alias_identity(row: pd.Series) -> str:
    """Build a stable identity for a strictly evidenced tournament-name alias."""
    parts = [
        str(row["tourney_date"]),
        "TOURNAMENT_ALIAS",
        str(row["round_normalized"]),
        str(row["winner_name_normalized"]),
        str(row["loser_name_normalized"]),
        str(row["score_normalized"]),
        str(row["surface_normalized"]),
    ]
    return " | ".join(parts)


def resolve_tournament_aliases(
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge only fully corroborated tournament-name aliases.

    A raw name difference alone is not enough to merge rows.  Automatic merging
    requires the same tournament week, round, winner, loser, normalized score,
    and surface, with all of those values present.  Candidate groups with
    conflicting or incomplete evidence remain distinct and are emitted for
    manual review.
    """
    required = {
        *ALIAS_CANDIDATE_COLUMNS,
        *ALIAS_EVIDENCE_COLUMNS,
        "tourney_name",
        "tourney_name_normalized",
        "round",
        "winner_name",
        "loser_name",
        "score",
        "surface",
        "provider_label",
        "source_file",
        "match_identity",
    }
    missing = sorted(required - set(matches.columns))
    if missing:
        raise ValueError(f"Alias resolution is missing required columns: {missing}")

    resolved = matches.copy()
    resolved["tournament_alias_resolution"] = "exact_tournament_name"
    report_rows: list[dict[str, object]] = []
    grouped = resolved.groupby(ALIAS_CANDIDATE_COLUMNS, dropna=False, sort=True)
    for _, group in grouped:
        tournament_variants = group["tourney_name_normalized"].replace("", pd.NA).dropna()
        if tournament_variants.nunique() < 2:
            continue

        evidence_complete = all(
            group[column].replace("", pd.NA).notna().all()
            for column in ALIAS_CANDIDATE_COLUMNS[1:] + ALIAS_EVIDENCE_COLUMNS
        )
        evidence_agrees = all(
            group[column].nunique(dropna=False) == 1
            for column in ALIAS_EVIDENCE_COLUMNS
        )
        report = {
            "tourney_date": group["tourney_date"].iloc[0],
            "round": group["round"].iloc[0],
            "winner_name": group["winner_name"].iloc[0],
            "loser_name": group["loser_name"].iloc[0],
            "score": group["score"].iloc[0],
            "surface": group["surface"].iloc[0],
            "tournament_name_variants": joined_unique(group["tourney_name"]),
            "tournament_name_normalized_variants": joined_unique(
                group["tourney_name_normalized"]
            ),
            "source_providers": joined_unique(group["provider_label"]),
            "source_files": joined_unique(group["source_file"]),
            "normalized_score_variants": joined_unique(group["score_normalized"]),
            "normalized_surface_variants": joined_unique(group["surface_normalized"]),
            "source_rows_before": int(len(group)),
        }
        if evidence_complete and evidence_agrees:
            group_identity = alias_identity(group.iloc[0])
            resolved.loc[group.index, "match_identity"] = group_identity
            resolved.loc[
                group.index, "tournament_alias_resolution"
            ] = "auto_merged_strict_alias"
            report.update(
                {
                    "action": "consolidated",
                    "records_removed": int(len(group) - 1),
                    "reason": (
                        "Same tournament week, round, winner, loser, normalized "
                        "score, and surface; only tournament-name aliases differ."
                    ),
                    "resolved_match_identity": group_identity,
                }
            )
        else:
            report.update(
                {
                    "action": "review_required",
                    "records_removed": 0,
                    "reason": (
                        "Tournament-name variants share week, round, and players, "
                        "but score or surface evidence conflicts or is incomplete."
                    ),
                    "resolved_match_identity": "",
                }
            )
        report_rows.append(report)

    report_columns = [
        "action",
        "reason",
        "tourney_date",
        "round",
        "winner_name",
        "loser_name",
        "score",
        "surface",
        "tournament_name_variants",
        "tournament_name_normalized_variants",
        "source_providers",
        "source_files",
        "normalized_score_variants",
        "normalized_surface_variants",
        "source_rows_before",
        "records_removed",
        "resolved_match_identity",
    ]
    report = pd.DataFrame(report_rows, columns=report_columns)
    return resolved, report


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
        "plus strict tournament-alias resolution when round, players, score, and "
        "surface agree; most complete record; TennisMyLife tie-break"
    )
    combined = combined.drop(
        columns=[
            "provider_tie_break",
            "tourney_name_normalized",
            "round_normalized",
            "winner_name_normalized",
            "loser_name_normalized",
            "score_normalized",
            "surface_normalized",
        ]
    )
    if combined["match_identity"].duplicated().any():
        raise ValueError("Combined output still contains duplicate match identities.")
    return combined.sort_values(
        ["source_year", "tourney_date", "tourney_name", "round", "winner_name", "loser_name"],
        kind="stable",
    ).reset_index(drop=True)


def build_combined_dataset(
    output_path: Path = OUTPUT_PATH,
    alias_report_path: Path = ALIAS_REPORT_PATH,
) -> pd.DataFrame:
    """Create the persisted canonical dataset from the unchanged raw inputs."""
    prepared = prepare_source_rows(load_source_rows())
    resolved, alias_report = resolve_tournament_aliases(prepared)
    combined = combine_source_rows(resolved)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    alias_report_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    alias_report.to_csv(alias_report_path, index=False)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output CSV path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--alias-report",
        type=Path,
        default=ALIAS_REPORT_PATH,
        help=(
            "CSV audit of automatic tournament-name alias merges and unresolved "
            f"candidates (default: {ALIAS_REPORT_PATH})"
        ),
    )
    args = parser.parse_args()
    combined = build_combined_dataset(args.output, args.alias_report)
    print(f"Wrote {len(combined):,} unique matches to {args.output}")
    print(
        "Cross-source duplicate identities consolidated: "
        f"{int(combined['is_cross_source_duplicate'].sum()):,}"
    )
    print("Retained provider counts:")
    print(combined["retained_provider"].value_counts().to_string())
    print(f"Wrote tournament-alias audit to {args.alias_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
