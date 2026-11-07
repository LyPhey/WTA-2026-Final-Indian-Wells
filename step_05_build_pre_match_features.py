"""Build time-aware, leakage-safe WTA pre-match features.

The source files only supply tournament-week dates, not dependable match dates
or start times. Every row without a verified completion timestamp therefore
uses only completed results from strictly earlier tournament weeks. Within such
a week, all player histories and Elo ratings are frozen until every result in
that week has been processed. Dashboard feedback may provide a verified
``match_completed_at_utc`` timestamp; those results become available only to
later timestamped live predictions.

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
FEATURE_VERSION = "v2_strict_prior_week_rolling_form_elo"
ROLLING_WINDOWS_WEEKS = (13, 26, 52)
RECENT_MATCH_WINDOW = 10
ELO_INITIAL_RATING = 1_500.0
ELO_K_FACTOR = 32.0
MATCH_METRICS = [
    "aces_per_match",
    "double_faults_per_match",
    "first_serve_won_pct",
    "second_serve_won_pct",
    "serve_points_won_pct",
    "match_minutes",
]
ROLLING_SUMMARY_FIELDS = [
    "matches",
    "wins",
    "win_rate",
    "surface_matches",
    "surface_wins",
    "surface_win_rate",
    *MATCH_METRICS,
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
    """Create a stable, accent-insensitive player key for historical state."""
    if pd.isna(value):
        return ""
    ascii_value = unicodedata.normalize("NFKD", str(value)).encode(
        "ascii", "ignore"
    ).decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()).strip()


def clean_category(value: object, fallback: str = "Unknown") -> str:
    """Return a non-empty category without turning missing values into zero."""
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def finite_number(value: object) -> float:
    """Return a finite float or NaN when a source value is unavailable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if math.isfinite(number) else np.nan


def safe_ratio(numerator: object, denominator: object) -> float:
    """Calculate a rate only when its denominator is available and positive."""
    numerator_value = finite_number(numerator)
    denominator_value = finite_number(denominator)
    if np.isnan(numerator_value) or np.isnan(denominator_value) or denominator_value <= 0:
        return np.nan
    return numerator_value / denominator_value


def difference(left: float, right: float) -> float:
    """Preserve missingness rather than replacing unavailable differences with zero."""
    if np.isnan(left) or np.isnan(right):
        return np.nan
    return left - right


def expected_score(player_rating: float, opponent_rating: float) -> float:
    """Return the standard Elo expected win probability."""
    return 1.0 / (1.0 + 10 ** ((opponent_rating - player_rating) / 400.0))


@dataclass(frozen=True)
class HistoricalMatch:
    """One completed result available after its tournament week ends."""

    match_date: pd.Timestamp
    surface: str
    won: bool
    metrics: dict[str, float]


@dataclass
class PlayerHistory:
    """Player state that is inspected only before the current tournament week."""

    matches: list[HistoricalMatch] = field(default_factory=list)
    recent_results: deque[int] = field(
        default_factory=lambda: deque(maxlen=RECENT_MATCH_WINDOW)
    )

    @staticmethod
    def summarize(records: list[HistoricalMatch], surface: str) -> dict[str, float]:
        """Aggregate form, surface form, and service metrics over given records."""
        surface_records = [record for record in records if record.surface == surface]
        summary = {
            "matches": float(len(records)),
            "wins": float(sum(record.won for record in records)),
            "win_rate": (
                sum(record.won for record in records) / len(records)
                if records
                else np.nan
            ),
            "surface_matches": float(len(surface_records)),
            "surface_wins": float(sum(record.won for record in surface_records)),
            "surface_win_rate": (
                sum(record.won for record in surface_records) / len(surface_records)
                if surface_records
                else np.nan
            ),
        }
        for metric in MATCH_METRICS:
            values = [record.metrics[metric] for record in records]
            valid = [value for value in values if not np.isnan(value)]
            summary[metric] = float(np.mean(valid)) if valid else np.nan
        return summary

    def snapshot(self, current_date: pd.Timestamp, surface: str) -> dict[str, float]:
        """Create rolling form features from matches strictly before current_date."""
        values: dict[str, float] = {
            "weeks_since_last_match": (
                (current_date - self.matches[-1].match_date).days / 7.0
                if self.matches
                else np.nan
            ),
            "recent_10_matches": float(len(self.recent_results)),
            "recent_10_win_rate": (
                sum(self.recent_results) / len(self.recent_results)
                if self.recent_results
                else np.nan
            ),
        }
        for weeks in ROLLING_WINDOWS_WEEKS:
            earliest_date = current_date - pd.Timedelta(weeks=weeks)
            window_records = [
                record for record in self.matches if record.match_date >= earliest_date
            ]
            summary = self.summarize(window_records, surface)
            for name, value in summary.items():
                values[f"prior_{weeks}w_{name}"] = value
        return values

    def add_completed_match(
        self,
        *,
        match_date: pd.Timestamp,
        surface: str,
        won: bool,
        metrics: dict[str, float],
    ) -> None:
        """Make a completed result available for only future tournament weeks."""
        self.matches.append(
            HistoricalMatch(
                match_date=match_date,
                surface=surface,
                won=won,
                metrics=metrics,
            )
        )
        self.recent_results.append(int(won))


def player_match_metrics(row: Any, prefix: str) -> dict[str, float]:
    """Derive post-match statistics for later historical windows only."""
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
    """Return the selected player's ranking and ranking points before this match."""
    return (
        finite_number(getattr(row, f"{side}_rank")),
        finite_number(getattr(row, f"{side}_rank_points")),
    )


def h2h_snapshot(
    counts: Counter[str], player_a_key: str
) -> tuple[float, float, float]:
    """Return prior meetings, Player A wins, and Player A's prior win rate."""
    matches = sum(counts.values())
    player_a_wins = counts[player_a_key]
    return (
        float(matches),
        float(player_a_wins),
        player_a_wins / matches if matches else np.nan,
    )


def prepare_completed_matches(source: pd.DataFrame) -> pd.DataFrame:
    """Keep valid results and preserve weekly fallback when no timestamp exists."""
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
    if "match_completed_at_utc" in prepared:
        completed_at = pd.to_datetime(
            prepared["match_completed_at_utc"], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        completed_at = pd.Series(pd.NaT, index=prepared.index, dtype="datetime64[ns]")
    # Standard provider rows retain their weekly timestamp. Only verified
    # dashboard feedback has enough timing information to update same-event
    # state for a later match.
    prepared["feature_available_at"] = completed_at.fillna(prepared["match_date"])
    prepared["winner_key"] = prepared["winner_name"].map(normalized_player_key)
    prepared["loser_key"] = prepared["loser_name"].map(normalized_player_key)
    eligible = prepared.loc[
        prepared["match_status"].eq("Completed")
        & prepared["match_date"].notna()
        & prepared["winner_key"].ne("")
        & prepared["loser_key"].ne("")
        & prepared["winner_key"].ne(prepared["loser_key"])
    ].copy()
    return eligible.sort_values(["feature_available_at", "match_identity"], kind="stable")


def feature_fields() -> list[str]:
    """Return player-level history field names emitted in every feature row."""
    fields = ["weeks_since_last_match", "recent_10_matches", "recent_10_win_rate"]
    for weeks in ROLLING_WINDOWS_WEEKS:
        fields.extend(f"prior_{weeks}w_{name}" for name in ROLLING_SUMMARY_FIELDS)
    return fields


def build_pre_match_features(source: pd.DataFrame) -> pd.DataFrame:
    """Build neutral Player A-vs-Player B features without outcome leakage."""
    matches = prepare_completed_matches(source)
    players: defaultdict[str, PlayerHistory] = defaultdict(PlayerHistory)
    head_to_head: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    surface_head_to_head: defaultdict[tuple[str, str, str], Counter[str]] = defaultdict(
        Counter
    )
    overall_elo: defaultdict[str, float] = defaultdict(lambda: ELO_INITIAL_RATING)
    surface_elo: defaultdict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(lambda: ELO_INITIAL_RATING)
    )
    history_fields = feature_fields()
    records: list[dict[str, object]] = []
    prior_available_at: pd.Timestamp | None = None

    for available_at, time_matches in matches.groupby("feature_available_at", sort=True):
        for row in time_matches.itertuples(index=False):
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
            player_a_history = players[player_a_key].snapshot(available_at, surface)
            player_b_history = players[player_b_key].snapshot(available_at, surface)
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

            player_a_elo = overall_elo[player_a_key]
            player_b_elo = overall_elo[player_b_key]
            player_a_surface_elo = surface_elo[surface][player_a_key]
            player_b_surface_elo = surface_elo[surface][player_b_key]
            record: dict[str, object] = {
                "feature_version": FEATURE_VERSION,
                "match_date": row.match_date.date().isoformat(),
                "source_match_identity": str(row.match_identity),
                "feature_available_at_utc": available_at.tz_localize("UTC").isoformat(),
                "feature_cutoff_date": (
                    prior_available_at.date().isoformat()
                    if prior_available_at is not None
                    else pd.NA
                ),
                "feature_cutoff_at_utc": (
                    prior_available_at.tz_localize("UTC").isoformat()
                    if prior_available_at is not None
                    else pd.NA
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
                "player_a_elo": player_a_elo,
                "player_b_elo": player_b_elo,
                "elo_advantage_player_a": player_a_elo - player_b_elo,
                "player_a_surface_elo": player_a_surface_elo,
                "player_b_surface_elo": player_b_surface_elo,
                "surface_elo_advantage_player_a": (
                    player_a_surface_elo - player_b_surface_elo
                ),
                "h2h_prior_matches": h2h_matches,
                "h2h_player_a_wins": h2h_player_a_wins,
                "h2h_player_a_win_rate": h2h_player_a_win_rate,
                "surface_h2h_prior_matches": surface_h2h_matches,
                "surface_h2h_player_a_wins": surface_h2h_player_a_wins,
                "surface_h2h_player_a_win_rate": surface_h2h_player_a_win_rate,
                "target_player_a_won": player_a_won,
            }
            for history_field in history_fields:
                player_a_value = player_a_history[history_field]
                player_b_value = player_b_history[history_field]
                record[f"player_a_{history_field}"] = player_a_value
                record[f"player_b_{history_field}"] = player_b_value
                record[f"{history_field}_advantage_player_a"] = difference(
                    player_a_value, player_b_value
                )
            records.append(record)

        overall_deltas: defaultdict[str, float] = defaultdict(float)
        surface_deltas: defaultdict[str, float] = defaultdict(float)
        for row in time_matches.itertuples(index=False):
            surface = clean_category(row.surface)
            winner_rating = overall_elo[row.winner_key]
            loser_rating = overall_elo[row.loser_key]
            winner_delta = ELO_K_FACTOR * (1.0 - expected_score(winner_rating, loser_rating))
            overall_deltas[row.winner_key] += winner_delta
            overall_deltas[row.loser_key] -= winner_delta

            winner_surface_rating = surface_elo[surface][row.winner_key]
            loser_surface_rating = surface_elo[surface][row.loser_key]
            winner_surface_delta = ELO_K_FACTOR * (
                1.0 - expected_score(winner_surface_rating, loser_surface_rating)
            )
            surface_deltas[(surface, row.winner_key)] += winner_surface_delta
            surface_deltas[(surface, row.loser_key)] -= winner_surface_delta

        for player_key, delta in overall_deltas.items():
            overall_elo[player_key] += delta
        for (surface, player_key), delta in surface_deltas.items():
            surface_elo[surface][player_key] += delta

        for row in time_matches.itertuples(index=False):
            surface = clean_category(row.surface)
            players[row.winner_key].add_completed_match(
                match_date=available_at,
                surface=surface,
                won=True,
                metrics=player_match_metrics(row, "w"),
            )
            players[row.loser_key].add_completed_match(
                match_date=available_at,
                surface=surface,
                won=False,
                metrics=player_match_metrics(row, "l"),
            )
            matchup_key = tuple(sorted((row.winner_key, row.loser_key)))
            head_to_head[matchup_key][row.winner_key] += 1
            surface_head_to_head[(surface, *matchup_key)][row.winner_key] += 1
        prior_available_at = available_at

    features = pd.DataFrame.from_records(records)
    validate_feature_table(features)
    return features


def validate_feature_table(features: pd.DataFrame) -> None:
    """Fail loudly when generated features violate their temporal controls."""
    required = {
        "match_date",
        "feature_available_at_utc",
        "feature_cutoff_date",
        "feature_cutoff_at_utc",
        "player_a",
        "player_b",
        "target_player_a_won",
        "elo_advantage_player_a",
        "prior_13w_win_rate_advantage_player_a",
        "prior_52w_surface_win_rate_advantage_player_a",
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
    available_at = pd.to_datetime(
        features["feature_available_at_utc"], errors="coerce", utc=True
    )
    cutoff_at = pd.to_datetime(features["feature_cutoff_at_utc"], errors="coerce", utc=True)
    if (cutoff_at.notna() & cutoff_at.ge(available_at)).any():
        raise ValueError("A row uses feature history from its own or a later time.")


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
        "same-week results and Elo updates are excluded."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
