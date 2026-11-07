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
import json
import shutil
from pathlib import Path
from statistics import NormalDist
import subprocess
import sys
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from step_05_build_pre_match_features import (
    build_pre_match_features,
    normalized_player_key,
)
from step_06_train_baseline_model import MODEL_FEATURES, TARGET_COLUMN, validate_input
from step_08_hyperparameter_tuning import MODEL_SPECS


PROJECT_DIR = Path(__file__).resolve().parent
CANONICAL_MATCHES_PATH = PROJECT_DIR / "data/processed/wta_matches_combined_deduplicated.csv"
FEATURES_PATH = PROJECT_DIR / "data/processed/wta_pre_match_features.csv"
CURRENT_TML_PATH = PROJECT_DIR / "data/raw/tennismylife/2026_wta.csv"
TML_RAW_DIR = CURRENT_TML_PATH.parent
DOWNLOAD_MANIFEST_PATH = PROJECT_DIR / "data/raw/download_manifest.csv"
FEEDBACK_RESULTS_PATH = PROJECT_DIR / "data/live_feedback/verified_completed_matches.csv"
SNAPSHOT_ROOT = PROJECT_DIR / "data/snapshots/daily_refresh"
HISTORY_START_YEAR = 2016
HISTORY_END_YEAR = 2026
LIVE_TRAINING_WINDOW_YEARS = 4
LIVE_PROBABILITY_INTERVAL_CONFIDENCE_LEVEL = 0.95
LIVE_CALIBRATION_NEIGHBOR_COUNT = 400
PREDICTION_TOURNAMENT_NAME = "Live WTA Finals prediction"
TUNING_REPORTS_DIR = PROJECT_DIR / "reports/hyperparameter_tuning"
MODEL_COMPARISON_PATH = TUNING_REPORTS_DIR / "nested_walk_forward_model_comparison.csv"
FEEDBACK_COLUMNS = [
    "feedback_id",
    "recorded_at_utc",
    "feedback_match_key",
    "tourney_date",
    "tourney_name",
    "tourney_level",
    "surface",
    "round",
    "winner_name",
    "loser_name",
    "score",
    "winner_rank",
    "loser_rank",
    "winner_rank_points",
    "loser_rank_points",
    "match_completed_at_utc",
    "match_status",
    "match_identity",
    "feedback_status",
]


@dataclass(frozen=True)
class DailyRefreshResult:
    """Paths and short logs from one dashboard-triggered daily refresh."""

    snapshot_dir: Path
    downloaded_files: int
    appended_rows: int
    merged_files: int
    command_logs: dict[str, str]


@dataclass(frozen=True)
class FeedbackRecordResult:
    """A verified, append-only result submitted from the dashboard."""

    feedback_id: str
    added: bool
    recorded_at_utc: str


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


def _as_utc_naive_timestamp(value: object, label: str) -> pd.Timestamp:
    """Validate an exact timestamp and return a timezone-free UTC value."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{label} must include a valid time zone.")
    return timestamp.tz_convert("UTC").tz_localize(None)


def _feedback_match_key(
    *,
    tournament_week_start: object,
    tournament_name: str,
    round_name: str,
    player_one: str,
    player_two: str,
) -> str:
    """Identify one scheduled singles match without depending on its winner."""
    week_start = pd.Timestamp(tournament_week_start).normalize()
    player_keys = sorted((normalized_player_key(player_one), normalized_player_key(player_two)))
    if pd.isna(week_start) or not all(player_keys):
        raise ValueError("Tournament week and both player names are required.")
    return " | ".join(
        [
            week_start.date().isoformat(),
            normalized_player_key(tournament_name),
            str(round_name).strip().upper(),
            *player_keys,
        ]
    )


def _optional_positive_number(value: object) -> float:
    """Keep valid entered rankings while preserving missing values as NaN."""
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) and number > 0 else np.nan


def record_completed_match_feedback(
    *,
    tournament_week_start: object,
    tournament_name: str,
    tourney_level: str,
    surface: str,
    round_name: str,
    winner_name: str,
    loser_name: str,
    score: str,
    match_completed_at_utc: object,
    winner_rank: float | None = None,
    loser_rank: float | None = None,
    winner_rank_points: float | None = None,
    loser_rank_points: float | None = None,
    feedback_path: Path = FEEDBACK_RESULTS_PATH,
) -> FeedbackRecordResult:
    """Append one confirmed result without overwriting the feedback ledger.

    This ledger is deliberately separate from downloaded provider files. It is
    available to a later, timestamped live forecast in the same event, but a
    duplicate or conflicting entry for the same scheduled match is rejected.
    """
    week_start = pd.Timestamp(tournament_week_start).normalize()
    completed_at = _as_utc_naive_timestamp(
        match_completed_at_utc, "Match completion time"
    )
    tournament_name = str(tournament_name).strip()
    round_name = str(round_name).strip().upper()
    surface = str(surface).strip()
    tourney_level = str(tourney_level).strip()
    winner_name = str(winner_name).strip()
    loser_name = str(loser_name).strip()
    score = str(score).strip()
    if pd.isna(week_start) or not tournament_name or not round_name or not surface or not tourney_level:
        raise ValueError("Tournament week, name, level, surface, and round are required.")
    if not winner_name or not loser_name or normalized_player_key(winner_name) == normalized_player_key(loser_name):
        raise ValueError("Enter two distinct players and select the actual winner.")
    if not score:
        raise ValueError("Enter the completed match score before saving feedback.")
    if completed_at < week_start:
        raise ValueError("Match completion cannot be before the tournament week starts.")

    match_key = _feedback_match_key(
        tournament_week_start=week_start,
        tournament_name=tournament_name,
        round_name=round_name,
        player_one=winner_name,
        player_two=loser_name,
    )
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        pd.read_csv(feedback_path, low_memory=False)
        if feedback_path.is_file()
        else pd.DataFrame(columns=FEEDBACK_COLUMNS)
    )
    if "feedback_match_key" in existing:
        conflicts = existing.loc[existing["feedback_match_key"].eq(match_key)]
    else:
        conflicts = pd.DataFrame()
    if not conflicts.empty:
        prior = conflicts.iloc[0]
        same_result = (
            str(prior.get("winner_name", "")).strip() == winner_name
            and str(prior.get("loser_name", "")).strip() == loser_name
            and str(prior.get("score", "")).strip() == score
            and str(prior.get("match_completed_at_utc", "")).strip()
            == completed_at.tz_localize("UTC").isoformat()
        )
        if same_result:
            return FeedbackRecordResult(
                feedback_id=str(prior["feedback_id"]),
                added=False,
                recorded_at_utc=str(prior["recorded_at_utc"]),
            )
        raise ValueError(
            "A different verified feedback record already exists for this match. "
            "The append-only ledger will not overwrite it."
        )

    recorded_at = pd.Timestamp.now(tz="UTC")
    feedback_id = f"feedback_{uuid4().hex}"
    record = {
        "feedback_id": feedback_id,
        "recorded_at_utc": recorded_at.isoformat(),
        "feedback_match_key": match_key,
        "tourney_date": week_start.strftime("%Y%m%d"),
        "tourney_name": tournament_name,
        "tourney_level": tourney_level,
        "surface": surface,
        "round": round_name,
        "winner_name": winner_name,
        "loser_name": loser_name,
        "score": score,
        "winner_rank": _optional_positive_number(winner_rank),
        "loser_rank": _optional_positive_number(loser_rank),
        "winner_rank_points": _optional_positive_number(winner_rank_points),
        "loser_rank_points": _optional_positive_number(loser_rank_points),
        "match_completed_at_utc": completed_at.tz_localize("UTC").isoformat(),
        "match_status": "Completed",
        "match_identity": feedback_id,
        "feedback_status": "verified",
    }
    merged = pd.concat(
        [existing.reindex(columns=FEEDBACK_COLUMNS), pd.DataFrame([record], columns=FEEDBACK_COLUMNS)],
        ignore_index=True,
    )
    temporary_path = feedback_path.with_suffix(feedback_path.suffix + ".tmp")
    merged.to_csv(temporary_path, index=False)
    temporary_path.replace(feedback_path)
    return FeedbackRecordResult(
        feedback_id=feedback_id, added=True, recorded_at_utc=recorded_at.isoformat()
    )


def load_verified_feedback(source: pd.DataFrame, feedback_path: Path = FEEDBACK_RESULTS_PATH) -> pd.DataFrame:
    """Add verified feedback rows to a canonical source without changing it on disk."""
    if not feedback_path.is_file():
        return source.copy()
    feedback = pd.read_csv(feedback_path, low_memory=False)
    required = set(FEEDBACK_COLUMNS)
    missing = sorted(required - set(feedback.columns))
    if missing:
        raise ValueError(f"Feedback ledger is missing required columns: {missing}")
    feedback = feedback.loc[feedback["feedback_status"].eq("verified")].copy()
    if feedback.empty:
        return source.copy()
    all_columns = list(dict.fromkeys([*source.columns, *feedback.columns]))
    return pd.concat(
        [source.reindex(columns=all_columns), feedback.reindex(columns=all_columns)],
        ignore_index=True,
        sort=False,
    )


def _source_available_before_match(
    source: pd.DataFrame,
    *,
    tournament_week_start: object,
    tournament_name: str,
    match_start_utc: object,
) -> tuple[pd.DataFrame, set[str]]:
    """Return prior-week history plus earlier verified results from this event only."""
    week_start = pd.Timestamp(tournament_week_start).normalize()
    match_start = _as_utc_naive_timestamp(match_start_utc, "Upcoming match start time")
    if pd.isna(week_start) or match_start < week_start:
        raise ValueError("Upcoming match start must be on or after the tournament week start.")
    prepared = source.copy()
    prepared["_week_start"] = pd.to_datetime(
        prepared["tourney_date"].astype("string"), format="%Y%m%d", errors="coerce"
    )
    base_history = prepared.loc[prepared["_week_start"].lt(week_start)].copy()
    completion_values = (
        prepared["match_completed_at_utc"]
        if "match_completed_at_utc" in prepared
        else pd.Series(pd.NaT, index=prepared.index, dtype="datetime64[ns]")
    )
    feedback_time = pd.to_datetime(
        completion_values, errors="coerce", utc=True
    ).dt.tz_localize(None)
    feedback_event = (
        prepared.get("feedback_status", pd.Series(index=prepared.index, dtype="object"))
        .eq("verified")
        & prepared["_week_start"].eq(week_start)
        & prepared["tourney_name"].map(normalized_player_key).eq(normalized_player_key(tournament_name))
        & feedback_time.lt(match_start)
    )
    event_feedback = prepared.loc[feedback_event].copy()
    feedback_ids = set(event_feedback["match_identity"].astype(str))
    selected = pd.concat([base_history, event_feedback], ignore_index=True, sort=False)
    selected = selected.drop(columns="_week_start")
    return selected, feedback_ids


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
    tournament_name: str,
    match_start_utc: object,
    surface: str,
    tourney_level: str,
    round_name: str,
    player_one_rank: float | None,
    player_two_rank: float | None,
    player_one_rank_points: float | None,
    player_two_rank_points: float | None,
) -> tuple[pd.DataFrame, str, str, pd.DataFrame, int]:
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
    selected_source, feedback_ids = _source_available_before_match(
        source,
        tournament_week_start=cutoff,
        tournament_name=tournament_name,
        match_start_utc=match_start_utc,
    )
    match_start = _as_utc_naive_timestamp(match_start_utc, "Upcoming match start time")
    if first_key < second_key:
        player_a, player_b = player_one, player_two
        rank_a, rank_b = player_one_rank, player_two_rank
        points_a, points_b = player_one_rank_points, player_two_rank_points
    else:
        player_a, player_b = player_two, player_one
        rank_a, rank_b = player_two_rank, player_one_rank
        points_a, points_b = player_two_rank_points, player_one_rank_points

    prediction_identity = (
        f"LIVE_PREDICTION | {match_start.tz_localize('UTC').isoformat()} | {first_key} | {second_key}"
    )
    synthetic = {column: np.nan for column in selected_source.columns}
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
            "match_identity": prediction_identity,
            "match_completed_at_utc": match_start.tz_localize("UTC").isoformat(),
        }
    )
    augmented = pd.concat(
        [selected_source, pd.DataFrame([synthetic])], ignore_index=True, sort=False
    )
    features = build_pre_match_features(augmented)
    prediction = features.loc[
        features["source_match_identity"].eq(prediction_identity)
    ]
    if len(prediction) != 1:
        raise RuntimeError("Could not create exactly one live pre-match feature row.")
    feedback_training_rows = features.loc[
        features["source_match_identity"].isin(feedback_ids)
    ].copy()
    return (
        prediction[MODEL_FEATURES],
        player_a,
        player_b,
        feedback_training_rows,
        int(len(feedback_training_rows)),
    )


def _fit_live_model(
    feature_path: Path,
    tournament_week_start: object,
    feedback_training_rows: pd.DataFrame | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit the best backtested candidate only through the safe cutoff."""
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
    historical_training_rows = int(len(training))
    if feedback_training_rows is not None and not feedback_training_rows.empty:
        feedback_training_rows = validate_input(feedback_training_rows)
        training = pd.concat([training, feedback_training_rows], ignore_index=True, sort=False)
    if training.empty:
        raise ValueError("No historical feature rows are available before this tournament week.")
    if training[TARGET_COLUMN].nunique() != 2:
        raise ValueError("The live training window does not contain both outcome classes.")
    model_name, model_params, selection_log_loss = _select_live_model()
    pipeline = MODEL_SPECS[model_name].pipeline_builder().set_params(**model_params)
    pipeline.fit(training[MODEL_FEATURES], training[TARGET_COLUMN])
    return pipeline, {
        "training_rows": int(len(training)),
        "historical_training_rows": historical_training_rows,
        "same_event_feedback_training_rows": int(len(training) - historical_training_rows),
        "training_start_date": training["match_date"].min().date().isoformat(),
        "training_end_date": training["match_date"].max().date().isoformat(),
        "feature_version": training["feature_version"].iloc[0],
        "model": f"live_{model_name}_selected_by_outer_log_loss",
        "model_name": model_name,
        "model_params": model_params,
        "model_selection_log_loss": selection_log_loss,
        "model_selection_method": "lowest chronological outer-fold log loss",
    }


def _select_live_model() -> tuple[str, dict[str, Any], float]:
    """Choose the lowest-log-loss tuned model with auditable interval evidence."""
    if not MODEL_COMPARISON_PATH.is_file():
        raise FileNotFoundError(
            "Model comparison is missing. Run step_08_hyperparameter_tuning.py before "
            "creating live predictions."
        )
    comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    required_columns = {"model", "log_loss"}
    if missing := sorted(required_columns - set(comparison.columns)):
        raise ValueError(f"Model comparison is missing required columns: {missing}")
    comparison["log_loss"] = pd.to_numeric(comparison["log_loss"], errors="coerce")
    eligible = comparison.loc[
        comparison["model"].isin(MODEL_SPECS) & comparison["log_loss"].notna()
    ].sort_values("log_loss", kind="stable")
    if eligible.empty:
        raise ValueError("No tuned live-prediction model is available in the comparison report.")

    model_name = str(eligible.iloc[0]["model"])
    report_path = TUNING_REPORTS_DIR / f"{model_name}_nested_walk_forward.json"
    predictions_path = TUNING_REPORTS_DIR / f"{model_name}_outer_fold_predictions.csv"
    if not report_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(
            f"Live prediction needs {report_path.name} and {predictions_path.name}. "
            "Run step_08_hyperparameter_tuning.py to create current tuning and "
            "out-of-fold calibration outputs."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outer_folds = report.get("outer_folds", [])
    if not outer_folds:
        raise ValueError(f"The tuning report for {model_name} has no outer folds.")
    latest_fold = max(outer_folds, key=lambda fold: int(fold["evaluation_year"]))
    model_params = latest_fold.get("best_params")
    if not isinstance(model_params, dict):
        raise ValueError(f"The tuning report for {model_name} has no selected parameters.")
    return model_name, model_params, float(eligible.iloc[0]["log_loss"])


def _calibrated_probability_interval(
    probability: float,
    model_name: str,
    *,
    confidence_level: float = LIVE_PROBABILITY_INTERVAL_CONFIDENCE_LEVEL,
    neighbor_count: int = LIVE_CALIBRATION_NEIGHBOR_COUNT,
) -> tuple[float, float, dict[str, float | int]]:
    """Return a model-agnostic Wilson interval from nearby out-of-fold forecasts."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("The probability-interval confidence level must be between 0 and 1.")
    if neighbor_count < 1:
        raise ValueError("The calibration-neighbor count must be at least 1.")
    predictions_path = TUNING_REPORTS_DIR / f"{model_name}_outer_fold_predictions.csv"
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Out-of-fold predictions are missing for {model_name}. Run "
            "step_08_hyperparameter_tuning.py to create them."
        )
    historical = pd.read_csv(
        predictions_path,
        usecols=["actual_player_a_won", "probability_player_a_wins"],
    )
    historical["actual_player_a_won"] = pd.to_numeric(
        historical["actual_player_a_won"], errors="coerce"
    )
    historical["probability_player_a_wins"] = pd.to_numeric(
        historical["probability_player_a_wins"], errors="coerce"
    )
    historical = historical.loc[
        historical["actual_player_a_won"].isin([0, 1])
        & historical["probability_player_a_wins"].between(0.0, 1.0)
    ].copy()
    if historical.empty:
        raise ValueError(f"Out-of-fold predictions for {model_name} contain no valid outcomes.")
    historical["probability_distance"] = (
        historical["probability_player_a_wins"] - probability
    ).abs()
    neighborhood = historical.nsmallest(neighbor_count, "probability_distance")
    sample_size = len(neighborhood)
    observed_rate = float(neighborhood["actual_player_a_won"].mean())
    z_score = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    denominator = 1.0 + (z_score**2 / sample_size)
    center = (observed_rate + z_score**2 / (2.0 * sample_size)) / denominator
    half_width = z_score * np.sqrt(
        (observed_rate * (1.0 - observed_rate) + z_score**2 / (4.0 * sample_size))
        / sample_size
    ) / denominator
    return float(center - half_width), float(center + half_width), {
        "calibration_neighbor_count": sample_size,
        "calibration_observed_win_rate": observed_rate,
        "calibration_min_probability": float(
            neighborhood["probability_player_a_wins"].min()
        ),
        "calibration_max_probability": float(
            neighborhood["probability_player_a_wins"].max()
        ),
    }


def predict_live_match(
    source: pd.DataFrame,
    *,
    player_one: str,
    player_two: str,
    tournament_week_start: object,
    tournament_name: str,
    match_start_utc: object,
    surface: str,
    tourney_level: str,
    round_name: str,
    player_one_rank: float | None,
    player_two_rank: float | None,
    player_one_rank_points: float | None,
    player_two_rank_points: float | None,
    feature_path: Path = FEATURES_PATH,
    feedback_path: Path = FEEDBACK_RESULTS_PATH,
) -> dict[str, Any]:
    """Return complementary win probabilities for one known upcoming matchup."""
    source_with_feedback = load_verified_feedback(source, feedback_path)
    (
        feature_row,
        player_a,
        player_b,
        feedback_training_rows,
        feedback_result_count,
    ) = _build_prediction_feature_row(
        source_with_feedback,
        player_one=player_one,
        player_two=player_two,
        tournament_week_start=tournament_week_start,
        tournament_name=tournament_name,
        match_start_utc=match_start_utc,
        surface=surface,
        tourney_level=tourney_level,
        round_name=round_name,
        player_one_rank=player_one_rank,
        player_two_rank=player_two_rank,
        player_one_rank_points=player_one_rank_points,
        player_two_rank_points=player_two_rank_points,
    )
    pipeline, training_metadata = _fit_live_model(
        feature_path, tournament_week_start, feedback_training_rows
    )
    probability_a = float(pipeline.predict_proba(feature_row)[0, 1])
    lower_a, upper_a, interval_metadata = _calibrated_probability_interval(
        probability_a, training_metadata["model_name"]
    )
    probability_one = probability_a if player_one == player_a else 1.0 - probability_a
    probability_two = 1.0 - probability_one
    if player_one == player_a:
        lower_one, upper_one = lower_a, upper_a
    else:
        lower_one, upper_one = 1.0 - upper_a, 1.0 - lower_a
    lower_two, upper_two = 1.0 - upper_one, 1.0 - lower_one
    if not np.isclose(probability_one + probability_two, 1.0):
        raise RuntimeError("Live probabilities do not sum to one.")
    return {
        "player_one": player_one,
        "player_two": player_two,
        "player_one_probability": probability_one,
        "player_two_probability": probability_two,
        "player_one_probability_interval": (lower_one, upper_one),
        "player_two_probability_interval": (lower_two, upper_two),
        "probability_interval_confidence_level": LIVE_PROBABILITY_INTERVAL_CONFIDENCE_LEVEL,
        "probability_interval_method": "nearest_out_of_fold_calibration_wilson",
        "internal_player_a": player_a,
        "internal_player_b": player_b,
        "tournament_week_start": pd.Timestamp(tournament_week_start).date().isoformat(),
        "tournament_name": tournament_name,
        "match_start_utc": _as_utc_naive_timestamp(
            match_start_utc, "Upcoming match start time"
        ).tz_localize("UTC").isoformat(),
        "same_event_feedback_result_count": feedback_result_count,
        "surface": surface,
        "tourney_level": tourney_level,
        "round": round_name,
        **training_metadata,
        **interval_metadata,
    }
