"""Build a leakage-safe, historical WTA pre-match feature table.

The raw match files only provide tournament-week dates, not dependable match
dates or start times. To avoid assuming an order that the data cannot support,
each match receives features from completed matches in strictly earlier
tournament weeks. Results from the same tournament week are never used.

Run from the project directory:
    python step_05_build_pre_match_features.py
"""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/processed/wta_matches_combined_deduplicated.csv")
OUTPUT_PATH = Path("data/processed/wta_pre_match_features.csv")
FEATURE_VERSION = "v1_strict_prior_tournament_week"
RECENT_WINDOW = 10
HISTORY_METRICS = [
    "aces_per_match",
    "double_faults_per_match",
    "first_serve_won_pct",
    "second_serve_won_pct",
    "serve_points_won_pct",
    "match_minutes",
]
HISTORY_FIELDS = [
    "prior_matches",
    "prior_wins",
    "prior_win_rate",
    "prior_surface_matches",
    "prior_surface_wins",
    "prior_surface_win_rate",
    "recent_10_matches",
    "recent_10_win_rate",
    *[f"prior_{metric}" for metric in HISTORY_METRICS],
]
FORBIDDEN_OUTPUT_PREFIXES = ("w_", "l_", "winner_", "loser_", "weather_")
FORBIDDEN_OUTPUT_COLUMNS = {
    "score",
    "minutes",
    "source_record_count",
    "source_providers",
    "retained_provider",
}


def normalized_player_key(value: object) -> str:
    """Create a stable, accent-insensitive player key for history tracking."""
    if pd.isna(value):
        return ""
    ascii_value = unicodedata.normalize("NFKD", str(value)).encode(
        "ascii", "ignore"
    ).decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()).strip()


def clean_category(value: object, fallback: str = "Unknown") -> str:
    """Return a non-empty categorical value without turning missing into zero."""
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def finite_number(value: object) -> float:
    """Return a finite float or NaN for unavailable source values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def safe_ratio(numerator: object, denominator: object) -> float:
    """Compute a rate only when the source supplies a valid positive denominator."""
    numerator_value = finite_number(numerator)
    denominator_value = finite_number(denominator)
    if np.isnan(numerator_value) or np.isnan(denominator_value) or denominator_value <= 0:
        return np.nan
    return numerator_value / denominator_value


def difference(left: float, right: float) -> float:
    """Return a difference without disguising an unavailable input as zero."""
    if np.isnan(left) or np.isnan(right):
        return np.nan
    return left - right


@dataclass
class PlayerHistory:
    """Completed-match history for one player before a tournament-week cutoff."""

    matches: int = 0
    wins: int = 0
    surface_matches: Counter[str] = field(default_factory=Counter)
    surface_wins: Counter[str] = field(default_factory=Counter)
    recent_results: deque[int] = field(
        default_factory=lambda: deque(maxlen=RECENT_WINDOW)
    )
    metric_sums: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    metric_counts: Counter[str] = field(default_factory=Counter)

    def snapshot(self, surface: str) -> dict[str, float]:
        """Return only values accumulated before the current tournament week."""
        recent_matches = len(self.recent_results)
        values = {
            "prior_matches": float(self.matches),
            "prior_wins": float(self.wins),
            "prior_win_rate": self.wins / self.matches if self.matches else np.nan,
            "prior_surface_matches": float(self.surface_matches[surface]),
            "prior_surface_wins": float(self.surface_wins[surface]),
            "prior_surface_win_rate": (
                self.surface_wins[surface] / self.surface_matches[surface]
                if self.surface_matches[surface]
                else np.nan
            ),
            "recent_10_matches": float(recent_matches),
            "recent_10_win_rate": (
                sum(self.recent_results) / recent_matches if recent_matches else np.nan
            ),
        }
        for metric in HISTORY_METRICS:
            count = self.metric_counts[metric]
            values[f"prior_{metric}"] = (
                self.metric_sums[metric] / count if count else np.nan
            )
        return values

    def add_completed_match(
        self, *, surface: str, won: bool, metrics: dict[str, float]
    ) -> None:
        """Update history after a completed match becomes available."""
        self.matches += 1
        self.wins += int(won)
        self.surface_matches[surface] += 1
        self.surface_wins[surface] += int(won)
        self.recent_results.append(int(won))
        for metric, value in metrics.items():
            if not np.isnan(value):
                self.metric_sums[metric] += value
                self.metric_counts[metric] += 1


def player_match_metrics(row: Any, prefix: str) -> dict[str, float]:
    """Calculate post-match player statistics for later history updates only."""
    serve_points = finite_number(getattr(row, f"{prefix}_svpt"))
    first_serves_in = finite_number(getattr(row, f"{prefix}_1stIn"))
    first_serves_won = finite_number(getattr(row, f"{prefix}_1stWon"))
    second_serves_won = finite_number(getattr(row, f"{prefix}_2ndWon"))
    return {
        "aces_per_match": finite_number(getattr(row, f"{prefix}_ace")),
        "double_faults_per_match": finite_number(getattr(row, f"{prefix}_df")),
        "first_serve_won_pct": safe_ratio(first_serves_won, first_serves_in),
        "second_serve_won_pct": safe_ratio(
            second_serves_won,
            serve_points - first_serves_in
            if not np.isnan(serve_points) and not np.isnan(first_serves_in)
            else np.nan,
        ),
        "serve_points_won_pct": safe_ratio(
            first_serves_won + second_serves_won
            if not np.isnan(first_serves_won) and not np.isnan(second_serves_won)
            else np.nan,
            serve_points,
        ),
        "match_minutes": finite_number(getattr(row, "minutes")),
    }


def extract_entering_values(row: Any, side: str) -> tuple[float, float]:
    """Return the selected player's ranking and points entering this match."""
    return (
        finite_number(getattr(row, f"{side}_rank")),
        finite_number(getattr(row, f"{side}_rank_points")),
    )


def h2h_snapshot(
    counts: Counter[str], player_a_key: str
) -> tuple[float, float, float]:
    """Return pre-match meetings, Player A wins, and Player A win rate."""
    matches = sum(counts.values())
    player_a_wins = counts[player_a_key]
    return (
        float(matches),
        float(player_a_wins),
        player_a_wins / matches if matches else np.nan,
    )


def prepare_completed_matches(source: pd.DataFrame) -> pd.DataFrame:
    """Keep only valid, completed records eligible for historical labels."""
    required = {
        "tourney_date",
        "winner_name",
        "loser_name",
        "match_status",
        "surface",
        "match_identity",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Canonical input is missing required columns: {missing}")

    prepared = source.copy()
    prepared["match_date"] = pd.to_datetime(
        prepared["tourney_date"].astype("string"), format="%Y%m%d", errors="coerce"
    )
    prepared["winner_key"] = prepared["winner_name"].map(normalized_player_key)
    prepared["loser_key"] = prepared["loser_name"].map(normalized_player_key)
    eligible = prepared.loc[
        prepared["match_status"].eq("Completed")
        & prepared["match_date"].notna()
        & prepared["winner_key"].ne("")
        & prepared["loser_key"].ne("")
        & prepared["winner_key"].ne(prepared["loser_key"])
    ].copy()
    return eligible.sort_values(["match_date", "match_identity"], kind="stable")


def build_pre_match_features(source: pd.DataFrame) -> pd.DataFrame:
    """Create neutral Player A-vs-Player B rows without outcome leakage."""
    matches = prepare_completed_matches(source)
    players: defaultdict[str, PlayerHistory] = defaultdict(PlayerHistory)
    head_to_head: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    surface_head_to_head: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(
        Counter
    )
    records: list[dict[str, object]] = []
    prior_week: pd.Timestamp | None = None

    for match_date, week_matches in matches.groupby("match_date", sort=True):
        for row in week_matches.itertuples(index=False):
            winner_key = row.winner_key
            loser_key = row.loser_key
            if winner_key < loser_key:
                player_a_key, player_b_key = winner_key, loser_key
                player_a_name, player_b_name = row.winner_name, row.loser_name
                player_a_side, player_b_side = "winner", "loser"
                player_a_won = 1
            else:
                player_a_key, player_b_key = loser_key, winner_key
                player_a_name, player_b_name = row.loser_name, row.winner_name
                player_a_side, player_b_side = "loser", "winner"
                player_a_won = 0

            surface = clean_category(row.surface)
            player_a_history = players[player_a_key].snapshot(surface)
            player_b_history = players[player_b_key].snapshot(surface)
            player_a_rank, player_a_rank_points = extract_entering_values(
                row, player_a_side
            )
            player_b_rank, player_b_rank_points = extract_entering_values(
                row, player_b_side
            )

            h2h_key = tuple(sorted((player_a_key, player_b_key)))
            surface_h2h_key = (surface, *h2h_key)
            h2h_matches, h2h_player_a_wins, h2h_player_a_win_rate = h2h_snapshot(
                head_to_head[h2h_key], player_a_key
            )
            (
                surface_h2h_matches,
                surface_h2h_player_a_wins,
                surface_h2h_player_a_win_rate,
            ) = h2h_snapshot(surface_head_to_head[surface_h2h_key], player_a_key)

            record: dict[str, object] = {
                "feature_version": FEATURE_VERSION,
                "match_date": match_date.date().isoformat(),
                "feature_cutoff_date": (
                    prior_week.date().isoformat() if prior_week is not None else pd.NA
                ),
                "tourney_name": clean_category(row.tourney_name),
                "surface": surface,
                "tourney_level": clean_category(row.tourney_level),
                "round": clean_category(row.round),
                "player_a": player_a_name,
                "player_b": player_b_name,
                "player_a_rank_entering": player_a_rank,
                "player_b_rank_entering": player_b_rank,
                "player_a_rank_points_entering": player_a_rank_points,
                "player_b_rank_points_entering": player_b_rank_points,
                "rank_advantage_player_a": difference(player_b_rank, player_a_rank),
                "rank_points_advantage_player_a": difference(
                    player_a_rank_points, player_b_rank_points
                ),
                "h2h_prior_matches": h2h_matches,
                "h2h_player_a_wins": h2h_player_a_wins,
                "h2h_player_a_win_rate": h2h_player_a_win_rate,
                "surface_h2h_prior_matches": surface_h2h_matches,
                "surface_h2h_player_a_wins": surface_h2h_player_a_wins,
                "surface_h2h_player_a_win_rate": surface_h2h_player_a_win_rate,
                "target_player_a_won": player_a_won,
            }
            for history_field in HISTORY_FIELDS:
                player_a_value = player_a_history[history_field]
                player_b_value = player_b_history[history_field]
                record[f"player_a_{history_field}"] = player_a_value
                record[f"player_b_{history_field}"] = player_b_value
                record[f"{history_field}_advantage_player_a"] = difference(
                    player_a_value, player_b_value
                )
            records.append(record)

        for row in week_matches.itertuples(index=False):
            surface = clean_category(row.surface)
            winner_metrics = player_match_metrics(row, "w")
            loser_metrics = player_match_metrics(row, "l")
            players[row.winner_key].add_completed_match(
                surface=surface, won=True, metrics=winner_metrics
            )
            players[row.loser_key].add_completed_match(
                surface=surface, won=False, metrics=loser_metrics
            )
            matchup_key = tuple(sorted((row.winner_key, row.loser_key)))
            head_to_head[matchup_key][row.winner_key] += 1
            surface_head_to_head[(surface, *matchup_key)][row.winner_key] += 1
        prior_week = match_date

    features = pd.DataFrame.from_records(records)
    validate_feature_table(features)
    return features


def validate_feature_table(features: pd.DataFrame) -> None:
    """Fail loudly if the generated table violates its leakage controls."""
    required = {
        "match_date",
        "feature_cutoff_date",
        "player_a",
        "player_b",
        "target_player_a_won",
        "rank_advantage_player_a",
    }
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Feature output is missing required columns: {missing}")
    forbidden = sorted(
        column
        for column in features.columns
        if column.startswith(FORBIDDEN_OUTPUT_PREFIXES)
        or column in FORBIDDEN_OUTPUT_COLUMNS
    )
    if forbidden:
        raise ValueError(f"Feature output contains leakage-prone columns: {forbidden}")
    if not features["target_player_a_won"].isin([0, 1]).all():
        raise ValueError("The target must contain only binary Player A outcomes.")

    match_dates = pd.to_datetime(features["match_date"], errors="coerce")
    cutoff_dates = pd.to_datetime(features["feature_cutoff_date"], errors="coerce")
    invalid_cutoffs = cutoff_dates.notna() & cutoff_dates.ge(match_dates)
    if invalid_cutoffs.any():
        raise ValueError("A feature row uses history from its own or a later week.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    source = pd.read_csv(args.input, low_memory=False)
    features = build_pre_match_features(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Wrote {len(features):,} completed-match feature rows to {args.output}")
    print(f"Columns: {len(features.columns):,}")
    print(
        "History policy: strictly earlier tournament weeks; "
        "same-week results are excluded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
