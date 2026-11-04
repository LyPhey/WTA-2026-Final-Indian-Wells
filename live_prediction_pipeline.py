"""Leakage-safe daily data refreshes and live WTA match-win predictions.

This module is used by the Streamlit dashboard. A refresh downloads the full
available TennisMyLife history into an intake area, appends only raw rows not
already retained locally, then rebuilds the canonical matches and historical
feature table. Predictions deliberately use only matches from tournament weeks
strictly before the selected tournament week.

The source data describes tournament weeks rather than dependable match days.
Consequently, a group-stage result is never used for another match in that same
week, even when the dashboard is refreshed daily.  That conservative rule avoids
accidentally using a result that was unavailable before the next match.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import shutil
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from step_05_build_pre_match_features import (
    build_pre_match_features,
    normalized_player_key,
)
from step_06_train_baseline_model import MODEL_FEATURES, TARGET_COLUMN, validate_input
from step_08_hyperparameter_tuning import build_logistic_regression_pipeline


PROJECT_DIR = Path(__file__).resolve().parent
CANONICAL_MATCHES_PATH = PROJECT_DIR / "data/processed/wta_matches_combined_deduplicated.csv"
FEATURES_PATH = PROJECT_DIR / "data/processed/wta_pre_match_features.csv"
CURRENT_TML_PATH = PROJECT_DIR / "data/raw/tennismylife/2026_wta.csv"
TML_RAW_DIR = CURRENT_TML_PATH.parent
DOWNLOAD_MANIFEST_PATH = PROJECT_DIR / "data/raw/download_manifest.csv"
SNAPSHOT_ROOT = PROJECT_DIR / "data/snapshots/daily_refresh"
HISTORY_START_YEAR = 2016
HISTORY_END_YEAR = 2026
LIVE_TRAINING_WINDOW_YEARS = 4
LIVE_LOGISTIC_PARAMS = {
    "classifier__C": 0.1,
    "classifier__class_weight": None,
}
PREDICTION_TOURNAMENT_NAME = "Live WTA Finals prediction"


@dataclass(frozen=True)
class DailyRefreshResult:
    """Paths and short logs from one dashboard-triggered daily refresh."""

    snapshot_dir: Path
    downloaded_files: int
    appended_rows: int
    merged_files: int
    command_logs: dict[str, str]


def _command_output(command: list[str], label: str) -> str:
    """Run a project pipeline command and surface a concise actionable error."""
    completed = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
    if completed.returncode:
        detail = output[-4_000:] if output else "No command output was captured."
        raise RuntimeError(f"{label} failed (exit {completed.returncode}).\n{detail}")
    return output


def _snapshot_file(source: Path, destination: Path) -> None:
    """Copy one existing source file without mutating the original."""
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _snapshot_directory(source: Path, destination: Path) -> None:
    """Copy the retained provider history when it exists."""
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)


def _append_new_csv_rows(retained_path: Path, incoming_path: Path) -> int:
    """Append only exact-new source rows, preserving all retained raw history."""
    incoming = pd.read_csv(incoming_path, low_memory=False)
    if not retained_path.exists():
        retained_path.parent.mkdir(parents=True, exist_ok=True)
        incoming.to_csv(retained_path, index=False)
        return int(len(incoming))

    retained = pd.read_csv(retained_path, low_memory=False)
    columns = list(dict.fromkeys([*retained.columns, *incoming.columns]))
    retained = retained.reindex(columns=columns)
    incoming = incoming.reindex(columns=columns)
    retained_keys = set(
        retained.fillna("<MISSING>").astype(str).agg("\x1f".join, axis=1)
    )
    incoming_keys = incoming.fillna("<MISSING>").astype(str).agg("\x1f".join, axis=1)
    additions = incoming.loc[~incoming_keys.isin(retained_keys)]
    if additions.empty:
        return 0

    merged = pd.concat([retained, additions], ignore_index=True)
    temporary_path = retained_path.with_suffix(retained_path.suffix + ".merge")
    merged.to_csv(temporary_path, index=False)
    temporary_path.replace(retained_path)
    return int(len(additions))


def _merge_downloaded_tennismylife_history(incoming_dir: Path) -> tuple[int, int]:
    """Merge the downloaded provider files into retained history without replacement."""
    appended_rows = 0
    merged_files = 0
    for incoming_path in sorted(incoming_dir.glob("*.csv")):
        appended_rows += _append_new_csv_rows(TML_RAW_DIR / incoming_path.name, incoming_path)
        merged_files += 1
    if not merged_files:
        raise RuntimeError("The full-history download did not produce any TennisMyLife CSV files.")
    return appended_rows, merged_files


def refresh_daily_data() -> DailyRefreshResult:
    """Merge a full TennisMyLife refresh into local history without data loss.

    The function intentionally does not refit any saved model artifact.  Live
    predictions fit an in-memory model through their requested cutoff instead.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = SNAPSHOT_ROOT / stamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    _snapshot_directory(TML_RAW_DIR, snapshot_dir / "before/tennismylife")
    _snapshot_file(DOWNLOAD_MANIFEST_PATH, snapshot_dir / "before/download_manifest.csv")
    incoming_root = snapshot_dir / "incoming"

    logs = {
        "download": _command_output(
            [
                sys.executable,
                "download_data.py",
                "--provider",
                "tennismylife",
                "--start-year",
                str(HISTORY_START_YEAR),
                "--end-year",
                str(HISTORY_END_YEAR),
                "--out",
                str(incoming_root),
            ],
            "Full-history data download",
        )
    }
    appended_rows, merged_files = _merge_downloaded_tennismylife_history(
        incoming_root / "tennismylife"
    )
    _snapshot_file(incoming_root / "download_manifest.csv", DOWNLOAD_MANIFEST_PATH)
    _snapshot_directory(TML_RAW_DIR, snapshot_dir / "after/tennismylife")
    _snapshot_file(DOWNLOAD_MANIFEST_PATH, snapshot_dir / "after/download_manifest.csv")
    logs["combine"] = _command_output(
        [sys.executable, "build_combined_dataset.py"], "Canonical dataset rebuild"
    )
    logs["features"] = _command_output(
        [sys.executable, "step_05_build_pre_match_features.py"],
        "Pre-match feature rebuild",
    )
    downloaded_files = logs["download"].count("[downloaded")
    return DailyRefreshResult(
        snapshot_dir=snapshot_dir,
        downloaded_files=downloaded_files,
        appended_rows=appended_rows,
        merged_files=merged_files,
        command_logs=logs,
    )


def data_freshness() -> dict[str, str | int | None]:
    """Return safe-to-display coverage details for retained TennisMyLife history."""
    paths = sorted(TML_RAW_DIR.glob("*_wta.csv"))
    if not paths:
        return {
            "rows": 0,
            "coverage": None,
            "latest_tournament_week": None,
            "file_modified_utc": None,
        }
    dates: list[pd.Series] = []
    rows = 0
    for path in paths:
        source = pd.read_csv(path, usecols=["tourney_date"], low_memory=False)
        rows += len(source)
        dates.append(pd.to_datetime(source["tourney_date"], format="%Y%m%d", errors="coerce"))
    all_dates = pd.concat(dates, ignore_index=True)
    usable_dates = all_dates.dropna()
    return {
        "rows": int(rows),
        "coverage": (
            f"{usable_dates.min().year}–{usable_dates.max().year}"
            if not usable_dates.empty
            else None
        ),
        "latest_tournament_week": (
            usable_dates.max().date().isoformat() if not usable_dates.empty else None
        ),
        "file_modified_utc": datetime.fromtimestamp(
            max(path.stat().st_mtime for path in paths), timezone.utc
        ).isoformat(timespec="seconds"),
    }


def load_canonical_matches(path: Path = CANONICAL_MATCHES_PATH) -> pd.DataFrame:
    """Load the canonical history used for player state and source rankings."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Canonical match file is missing: {path}. Run build_combined_dataset.py first."
        )
    source = pd.read_csv(path, low_memory=False)
    required = {
        "tourney_date",
        "winner_name",
        "loser_name",
        "winner_rank",
        "loser_rank",
        "winner_rank_points",
        "loser_rank_points",
        "match_status",
        "match_identity",
        "surface",
        "round",
        "tourney_level",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Canonical match file is missing columns: {missing}")
    return source


def available_players(source: pd.DataFrame) -> list[str]:
    """Return the sorted player names that can be selected in the dashboard."""
    names = set(source["winner_name"].dropna().astype(str)) | set(
        source["loser_name"].dropna().astype(str)
    )
    return sorted(name for name in names if normalized_player_key(name))


def latest_known_ranking(
    source: pd.DataFrame,
    player_name: str,
    tournament_week_start: object,
) -> tuple[float | None, float | None, str | None]:
    """Return the latest source rank and points strictly before a tournament week."""
    cutoff = pd.Timestamp(tournament_week_start).normalize()
    matches = source.copy()
    matches["_week"] = pd.to_datetime(
        matches["tourney_date"].astype("string"), format="%Y%m%d", errors="coerce"
    )
    matches = matches.loc[
        matches["match_status"].eq("Completed") & matches["_week"].lt(cutoff)
    ].copy()
    key = normalized_player_key(player_name)
    candidates: list[pd.DataFrame] = []
    winner = matches.loc[
        matches["winner_name"].map(normalized_player_key).eq(key),
        ["_week", "winner_rank", "winner_rank_points"],
    ].rename(
        columns={"winner_rank": "rank", "winner_rank_points": "rank_points"}
    )
    loser = matches.loc[
        matches["loser_name"].map(normalized_player_key).eq(key),
        ["_week", "loser_rank", "loser_rank_points"],
    ].rename(columns={"loser_rank": "rank", "loser_rank_points": "rank_points"})
    candidates.extend([winner, loser])
    history = pd.concat(candidates, ignore_index=True)
    if history.empty:
        return None, None, None
    history["rank"] = pd.to_numeric(history["rank"], errors="coerce")
    history["rank_points"] = pd.to_numeric(history["rank_points"], errors="coerce")
    history = history.loc[(history["rank"] > 0) | (history["rank_points"] > 0)]
    if history.empty:
        return None, None, None
    latest = history.sort_values("_week", kind="stable").iloc[-1]
    rank = float(latest["rank"]) if pd.notna(latest["rank"]) and latest["rank"] > 0 else None
    points = (
        float(latest["rank_points"])
        if pd.notna(latest["rank_points"]) and latest["rank_points"] > 0
        else None
    )
    return rank, points, pd.Timestamp(latest["_week"]).date().isoformat()


def _build_prediction_feature_row(
    source: pd.DataFrame,
    *,
    player_one: str,
    player_two: str,
    tournament_week_start: object,
    surface: str,
    tourney_level: str,
    round_name: str,
    player_one_rank: float | None,
    player_two_rank: float | None,
    player_one_rank_points: float | None,
    player_two_rank_points: float | None,
) -> tuple[pd.DataFrame, str, str]:
    """Use the established step-05 builder to make one synthetic pre-match row.

    The synthetic row is never written to disk.  Its arbitrary winner label is
    used only because the historical builder requires a completed row; all model
    features are created before that label is read into history.
    """
    first_key = normalized_player_key(player_one)
    second_key = normalized_player_key(player_two)
    if not first_key or not second_key or first_key == second_key:
        raise ValueError("Choose two distinct, valid player names.")
    cutoff = pd.Timestamp(tournament_week_start).normalize()
    if pd.isna(cutoff):
        raise ValueError("Tournament-week start must be a valid date.")
    if first_key < second_key:
        player_a, player_b = player_one, player_two
        rank_a, rank_b = player_one_rank, player_two_rank
        points_a, points_b = player_one_rank_points, player_two_rank_points
    else:
        player_a, player_b = player_two, player_one
        rank_a, rank_b = player_two_rank, player_one_rank
        points_a, points_b = player_two_rank_points, player_one_rank_points

    synthetic = {column: np.nan for column in source.columns}
    synthetic.update(
        {
            "tourney_date": cutoff.strftime("%Y%m%d"),
            "tourney_name": PREDICTION_TOURNAMENT_NAME,
            "surface": surface,
            "tourney_level": tourney_level,
            "round": round_name,
            "winner_name": player_a,
            "loser_name": player_b,
            "winner_rank": rank_a,
            "loser_rank": rank_b,
            "winner_rank_points": points_a,
            "loser_rank_points": points_b,
            "match_status": "Completed",
            "match_identity": (
                f"LIVE_PREDICTION | {cutoff.date().isoformat()} | {first_key} | {second_key}"
            ),
        }
    )
    augmented = pd.concat([source, pd.DataFrame([synthetic])], ignore_index=True, sort=False)
    features = build_pre_match_features(augmented)
    prediction = features.loc[
        features["tourney_name"].eq(PREDICTION_TOURNAMENT_NAME)
        & features["match_date"].eq(cutoff.date().isoformat())
    ]
    if len(prediction) != 1:
        raise RuntimeError("Could not create exactly one live pre-match feature row.")
    return prediction[MODEL_FEATURES], player_a, player_b


def _fit_live_logistic_model(
    feature_path: Path,
    tournament_week_start: object,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit the selected logistic configuration only through the safe cutoff."""
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"Feature table is missing: {feature_path}. Run step_05_build_pre_match_features.py first."
        )
    cutoff = pd.Timestamp(tournament_week_start).normalize()
    features = validate_input(pd.read_csv(feature_path, low_memory=False))
    training_start = cutoff - pd.DateOffset(years=LIVE_TRAINING_WINDOW_YEARS)
    training = features.loc[
        features["match_date"].between(training_start, cutoff - pd.Timedelta(days=1))
    ].copy()
    if training.empty:
        raise ValueError("No historical feature rows are available before this tournament week.")
    if training[TARGET_COLUMN].nunique() != 2:
        raise ValueError("The live training window does not contain both outcome classes.")
    pipeline = build_logistic_regression_pipeline().set_params(**LIVE_LOGISTIC_PARAMS)
    pipeline.fit(training[MODEL_FEATURES], training[TARGET_COLUMN])
    return pipeline, {
        "training_rows": int(len(training)),
        "training_start_date": training["match_date"].min().date().isoformat(),
        "training_end_date": training["match_date"].max().date().isoformat(),
        "feature_version": training["feature_version"].iloc[0],
    }


def predict_live_match(
    source: pd.DataFrame,
    *,
    player_one: str,
    player_two: str,
    tournament_week_start: object,
    surface: str,
    tourney_level: str,
    round_name: str,
    player_one_rank: float | None,
    player_two_rank: float | None,
    player_one_rank_points: float | None,
    player_two_rank_points: float | None,
    feature_path: Path = FEATURES_PATH,
) -> dict[str, Any]:
    """Return complementary win probabilities for one known upcoming matchup."""
    feature_row, player_a, player_b = _build_prediction_feature_row(
        source,
        player_one=player_one,
        player_two=player_two,
        tournament_week_start=tournament_week_start,
        surface=surface,
        tourney_level=tourney_level,
        round_name=round_name,
        player_one_rank=player_one_rank,
        player_two_rank=player_two_rank,
        player_one_rank_points=player_one_rank_points,
        player_two_rank_points=player_two_rank_points,
    )
    pipeline, training_metadata = _fit_live_logistic_model(
        feature_path, tournament_week_start
    )
    probability_a = float(pipeline.predict_proba(feature_row)[0, 1])
    probability_one = probability_a if player_one == player_a else 1.0 - probability_a
    probability_two = 1.0 - probability_one
    if not np.isclose(probability_one + probability_two, 1.0):
        raise RuntimeError("Live probabilities do not sum to one.")
    return {
        "player_one": player_one,
        "player_two": player_two,
        "player_one_probability": probability_one,
        "player_two_probability": probability_two,
        "internal_player_a": player_a,
        "internal_player_b": player_b,
        "tournament_week_start": pd.Timestamp(tournament_week_start).date().isoformat(),
        "surface": surface,
        "tourney_level": tourney_level,
        "round": round_name,
        "model": "live_logistic_regression_tuned_v1",
        "model_params": LIVE_LOGISTIC_PARAMS,
        **training_metadata,
    }
