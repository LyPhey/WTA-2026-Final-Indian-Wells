"""Nested chronological hyperparameter tuning for the WTA winner models.

For every outer evaluation year, this script trains on only the immediately
preceding calendar years. Inside that outer training window, GridSearchCV uses
recent fixed-duration, tournament-week-aware validation splits to select
hyperparameters.
The outer year is never used to select a configuration.

The default grids are intentionally small.  Extend MODEL_SPECS with another
ModelSpec rather than adding a one-off tuning loop for each future model.

Run from the project directory:
    python step_08_hyperparameter_tuning.py

For a quicker smoke run of logistic regression only:
    python step_08_hyperparameter_tuning.py --models logistic_regression \\
        --first-evaluation-year 2025 --last-evaluation-year 2025 --inner-splits 2 \\
        --inner-validation-weeks 13
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


from step_06_train_baseline_model import (
    CATEGORICAL_FEATURES,
    INPUT_PATH,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    calculate_metrics,
    validate_input,
)


DEFAULT_REPORTS_DIR = Path("reports/hyperparameter_tuning")
DEFAULT_MODELS_DIR = Path("models/hyperparameter_tuning")
DEFAULT_FIRST_EVALUATION_YEAR = 2020
DEFAULT_LAST_EVALUATION_YEAR = 2025
DEFAULT_TRAIN_WINDOW_YEARS = 4
DEFAULT_INNER_SPLITS = 3
DEFAULT_INNER_VALIDATION_WEEKS = 26
DEFAULT_CALIBRATION_BINS = 10


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to tune one model in the shared evaluation framework."""

    name: str
    description: str
    pipeline_builder: Callable[[], Pipeline]
    param_grid: dict[str, list[Any]]


def build_logistic_regression_pipeline() -> Pipeline:
    """Return the step-06 preprocessing plus a logistic-regression classifier."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
                        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=2_000, random_state=42),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """Return leakage-safe preprocessing plus a CPU-friendly random forest.

    The forest itself uses one worker.  GridSearchCV owns parallelism, which
    avoids oversubscribing the machine by running many multi-threaded forests.
    """
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
                        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
                    ]
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    n_jobs=1,
                    random_state=42,
                ),
            ),
        ]
    )

def build_xgbost_pipeline() -> Pipeline:
    """Return leakage-safe preprocessing plus a CPU-friendly XGBoost.
    """
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
                        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
                    ]
                ),
            ),
            (
                "classifier",
                XGBClassifier(
                    objective='binary:logistic',
                    learning_rate=0.001,
                    n_estimators=1000,
                ),
            ),
        ]
    )
MODEL_SPECS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        name="logistic_regression",
        description="Regularized linear probability baseline using step-06 features.",
        pipeline_builder=build_logistic_regression_pipeline,
        param_grid={
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"],
        },
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        description="Nonlinear tree ensemble using the same explicit safe features.",
        pipeline_builder=build_random_forest_pipeline,
        param_grid={
            "classifier__max_depth": [None, 12],
            "classifier__min_samples_leaf": [2, 5],
            "classifier__max_features": ["sqrt", 0.5],
        },
    ),
    "xgboost": ModelSpec(
        name="xgboost",
        description="Nonlinear tree ensemble using the same explicit safe features.",
        pipeline_builder=build_xgbost_pipeline,
        param_grid={
            "classifier__max_depth": [2, 3],
        },
    ),
}


def load_features(path: Path) -> pd.DataFrame:
    """Load the shared, leakage-safe feature table and validate its schema."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Feature table does not exist: {path}. Run step_05_build_pre_match_features.py first."
        )
    prepared = validate_input(pd.read_csv(path, low_memory=False))
    prepared = prepared.sort_values("match_date", kind="stable").reset_index(drop=True)
    if prepared.empty:
        raise ValueError("The feature table is empty.")
    if prepared["match_date"].duplicated().all():
        raise ValueError("The feature table must contain more than one tournament week.")
    return prepared


def annual_outer_fold(
    features: pd.DataFrame, evaluation_year: int, train_window_years: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a strict rolling-history training window and later held-out year."""
    evaluation_start = pd.Timestamp(year=evaluation_year, month=1, day=1)
    evaluation_end = pd.Timestamp(year=evaluation_year, month=12, day=31)
    training_start = evaluation_start - pd.DateOffset(years=train_window_years)
    training_end = evaluation_start - pd.Timedelta(days=1)
    training = features.loc[
        features["match_date"].between(training_start, training_end)
    ].copy().reset_index(drop=True)
    evaluation = features.loc[
        features["match_date"].between(evaluation_start, evaluation_end)
    ].copy().reset_index(drop=True)
    if training.empty or evaluation.empty:
        raise ValueError(
            f"Outer fold {evaluation_year} has an empty training or evaluation set."
        )
    if training["match_date"].max() >= evaluation["match_date"].min():
        raise ValueError(f"Outer fold {evaluation_year} is not strictly time ordered.")
    if training[TARGET_COLUMN].nunique() != 2:
        raise ValueError(f"Outer fold {evaluation_year} training data lacks an outcome class.")
    if evaluation[TARGET_COLUMN].nunique() != 2:
        raise ValueError(f"Outer fold {evaluation_year} evaluation data lacks an outcome class.")
    return training, evaluation


def rolling_week_splits(
    training: pd.DataFrame, inner_splits: int, validation_weeks: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, str]]]:
    """Create recent fixed-duration validation folds without splitting tournament weeks.

    Each validation fold covers the same calendar duration, ending with the
    most recent available training week. Its fitting rows are strictly earlier
    tournament weeks and expand through history. This avoids the uneven,
    multi-season validation blocks produced by an equal-date-count split.
    """
    if inner_splits < 2:
        raise ValueError("--inner-splits must be at least 2.")
    if validation_weeks < 1:
        raise ValueError("--inner-validation-weeks must be at least 1.")
    unique_dates = pd.Index(training["match_date"].drop_duplicates().sort_values())
    if len(unique_dates) < inner_splits + 1:
        raise ValueError(
            f"Need at least {inner_splits + 1} distinct tournament weeks for "
            f"{inner_splits} inner splits; found {len(unique_dates)}."
        )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    metadata: list[dict[str, str]] = []
    dates = training["match_date"]
    last_week_end = pd.Timestamp(unique_dates.max()) + pd.Timedelta(days=7)
    validation_span = pd.Timedelta(weeks=validation_weeks)
    for split_number in range(1, inner_splits + 1):
        windows_remaining = inner_splits - split_number
        validation_end_exclusive = last_week_end - validation_span * windows_remaining
        validation_start = validation_end_exclusive - validation_span
        train_index = np.flatnonzero((dates < validation_start).to_numpy())
        validation_index = np.flatnonzero(
            ((dates >= validation_start) & (dates < validation_end_exclusive)).to_numpy()
        )
        if not len(train_index) or not len(validation_index):
            raise ValueError(
                f"Inner split {split_number} is empty for a {validation_weeks}-week "
                "validation window. Reduce --inner-splits or --inner-validation-weeks."
            )
        if dates.iloc[train_index].max() >= dates.iloc[validation_index].min():
            raise ValueError(f"Inner split {split_number} is not strictly time ordered.")
        if training.iloc[train_index][TARGET_COLUMN].nunique() != 2:
            raise ValueError(f"Inner split {split_number} training data lacks an outcome class.")
        if training.iloc[validation_index][TARGET_COLUMN].nunique() != 2:
            raise ValueError(f"Inner split {split_number} validation data lacks an outcome class.")
        splits.append((train_index, validation_index))
        metadata.append(
            {
                "inner_split": str(split_number),
                "training_start_date": dates.iloc[train_index].min().date().isoformat(),
                "training_end_date": dates.iloc[train_index].max().date().isoformat(),
                "validation_start_date": dates.iloc[validation_index].min().date().isoformat(),
                "validation_end_date": dates.iloc[validation_index].max().date().isoformat(),
                "validation_weeks": str(validation_weeks),
            }
        )
    return splits, metadata


def calibration_table(
    predictions: pd.DataFrame, calibration_bins: int
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Summarize observed outcomes against predicted probabilities by fixed bins."""
    if calibration_bins < 2:
        raise ValueError("--calibration-bins must be at least 2.")
    probabilities = predictions["probability_player_a_wins"].astype(float)
    outcomes = predictions["actual_player_a_won"].astype(float)
    if probabilities.isna().any() or outcomes.isna().any():
        raise ValueError("Out-of-fold predictions cannot contain missing values.")
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    bin_codes = pd.cut(
        probabilities, bins=edges, include_lowest=True, labels=False
    ).astype("Int64")
    rows: list[dict[str, float | int]] = []
    for bin_number in range(calibration_bins):
        in_bin = bin_codes.eq(bin_number)
        count = int(in_bin.sum())
        row: dict[str, float | int] = {
            "bin": bin_number + 1,
            "bin_lower": float(edges[bin_number]),
            "bin_upper": float(edges[bin_number + 1]),
            "match_count": count,
            "mean_predicted_probability": np.nan,
            "observed_win_rate": np.nan,
            "absolute_calibration_gap": np.nan,
        }
        if count:
            mean_prediction = float(probabilities.loc[in_bin].mean())
            observed_rate = float(outcomes.loc[in_bin].mean())
            row.update(
                {
                    "mean_predicted_probability": mean_prediction,
                    "observed_win_rate": observed_rate,
                    "absolute_calibration_gap": abs(mean_prediction - observed_rate),
                }
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    non_empty = table.loc[table["match_count"].gt(0)]
    expected_calibration_error = float(
        (
            non_empty["absolute_calibration_gap"] * non_empty["match_count"]
        ).sum()
        / len(predictions)
    )
    return table, {
        "calibration_bins": calibration_bins,
        "out_of_fold_predictions": int(len(predictions)),
        "expected_calibration_error": round(expected_calibration_error, 6),
        "maximum_calibration_error": round(
            float(non_empty["absolute_calibration_gap"].max()), 6
        ),
    }


def reliability_figure(calibration: pd.DataFrame, model_name: str) -> go.Figure:
    """Create an interactive reliability diagram from non-empty calibration bins."""
    visible_bins = calibration.loc[calibration["match_count"].gt(0)]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
            line={"color": "#94a3b8", "dash": "dash"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=visible_bins["mean_predicted_probability"],
            y=visible_bins["observed_win_rate"],
            mode="lines+markers",
            name="Out-of-fold predictions",
            marker={"size": 9, "color": "#2563eb"},
            customdata=visible_bins[["bin_lower", "bin_upper", "match_count"]],
            hovertemplate=(
                "Predicted: %{x:.1%}<br>Observed: %{y:.1%}<br>"
                "Bin: %{customdata[0]:.0%}–%{customdata[1]:.0%}<br>"
                "Matches: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=f"{model_name}: out-of-fold reliability",
        xaxis={"title": "Mean predicted Player A win probability", "range": [0, 1], "tickformat": ".0%"},
        yaxis={"title": "Observed Player A win rate", "range": [0, 1], "tickformat": ".0%"},
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
    )
    return figure


def json_default(value: object) -> object:
    """Convert NumPy scalars in tuning metadata to standard JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def grid_results_table(
    search: GridSearchCV,
    model_name: str,
    evaluation_year: int,
) -> pd.DataFrame:
    """Make GridSearchCV's useful results easy to inspect in a CSV."""
    results = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame(
        {
            "model": model_name,
            "evaluation_year": evaluation_year,
            "candidate_rank": results["rank_test_score"].astype(int),
            "mean_inner_log_loss": -results["mean_test_score"],
            "std_inner_log_loss": results["std_test_score"],
            "mean_fit_seconds": results["mean_fit_time"],
            "parameters": results["params"].map(
                lambda params: json.dumps(params, sort_keys=True, default=json_default)
            ),
        }
    )
    return table.sort_values(["candidate_rank", "mean_inner_log_loss"]).reset_index(drop=True)


def nested_walk_forward_tuning(
    features: pd.DataFrame,
    spec: ModelSpec,
    *,
    first_evaluation_year: int,
    last_evaluation_year: int,
    train_window_years: int,
    inner_splits: int,
    inner_validation_weeks: int,
    calibration_bins: int,
    n_jobs: int,
) -> tuple[dict[str, object], dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune within each outer training fold, then score the untouched outer year."""
    if train_window_years < 1:
        raise ValueError("--train-window-years must be at least 1.")
    if last_evaluation_year < first_evaluation_year:
        raise ValueError("The final evaluation year must not precede the first year.")
    if last_evaluation_year >= 2026:
        raise ValueError("Keep 2026 out of tuning and evaluation.")
    if n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero; use -1 for all available cores.")

    fold_records: list[dict[str, object]] = []
    all_grid_results: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    latest_estimator: Pipeline | None = None
    latest_record: dict[str, object] | None = None

    for evaluation_year in range(first_evaluation_year, last_evaluation_year + 1):
        training, evaluation = annual_outer_fold(
            features, evaluation_year, train_window_years
        )
        inner_cv, inner_windows = rolling_week_splits(
            training, inner_splits, inner_validation_weeks
        )
        search = GridSearchCV(
            estimator=spec.pipeline_builder(),
            param_grid=spec.param_grid,
            scoring="neg_log_loss",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(training[MODEL_FEATURES], training[TARGET_COLUMN])
        probability = pd.Series(
            search.predict_proba(evaluation[MODEL_FEATURES])[:, 1],
            index=evaluation.index,
            name="probability_player_a_wins",
        )
        metrics = calculate_metrics(evaluation[TARGET_COLUMN], probability)
        record: dict[str, object] = {
            "evaluation_year": evaluation_year,
            "training_rows": int(len(training)),
            "training_start_date": training["match_date"].min().date().isoformat(),
            "training_end_date": training["match_date"].max().date().isoformat(),
            "evaluation_rows": int(len(evaluation)),
            "evaluation_start_date": evaluation["match_date"].min().date().isoformat(),
            "evaluation_end_date": evaluation["match_date"].max().date().isoformat(),
            "best_inner_log_loss": round(float(-search.best_score_), 6),
            "best_params": search.best_params_,
            "inner_validation_windows": inner_windows,
            **metrics,
        }
        fold_records.append(record)
        all_grid_results.append(grid_results_table(search, spec.name, evaluation_year))
        prediction_context_columns = [
            column
            for column in (
                "match_date",
                "feature_cutoff_date",
                "tourney_name",
                "player_a",
                "player_b",
                "surface",
                "tourney_level",
                "round",
            )
            if column in evaluation.columns
        ]
        fold_predictions = evaluation[prediction_context_columns].copy()
        fold_predictions.insert(0, "evaluation_year", evaluation_year)
        fold_predictions["actual_player_a_won"] = evaluation[TARGET_COLUMN]
        fold_predictions["probability_player_a_wins"] = probability
        all_predictions.append(fold_predictions)
        latest_estimator = search.best_estimator_
        latest_record = record
        print(
            f"{spec.name}, outer {evaluation_year}: "
            f"inner log loss {-search.best_score_:.3f}; "
            f"outer log loss {metrics['log_loss']:.3f}; "
            f"params {search.best_params_}"
        )

    if latest_estimator is None or latest_record is None:
        raise ValueError("No outer tuning folds were created.")
    predictions = pd.concat(all_predictions, ignore_index=True)
    overall_metrics = calculate_metrics(
        predictions["actual_player_a_won"], predictions["probability_player_a_wins"]
    )
    calibration, calibration_summary = calibration_table(predictions, calibration_bins)
    fold_metrics = pd.DataFrame(fold_records)
    mean_fold_metrics = {
        metric: round(float(fold_metrics[metric].mean()), 6) for metric in overall_metrics
    }
    artifact = {
        "pipeline": latest_estimator,
        "model": spec.name,
        "model_features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_version": features["feature_version"].iloc[0],
        "purpose": "latest nested walk-forward fold; not a match-day production model",
        "training_window_years": train_window_years,
        "inner_splits": inner_splits,
        "inner_validation_weeks": inner_validation_weeks,
        "calibration_bins": calibration_bins,
        "evaluation_year": latest_record["evaluation_year"],
        "best_params": latest_record["best_params"],
        "training_start_date": latest_record["training_start_date"],
        "training_end_date": latest_record["training_end_date"],
    }
    report = {
        "model": spec.name,
        "description": spec.description,
        "backtest_method": "nested annual rolling-window walk-forward",
        "tuning_scoring": "log_loss",
        "feature_version": artifact["feature_version"],
        "feature_count": len(MODEL_FEATURES),
        "training_window_years": train_window_years,
        "inner_splits": inner_splits,
        "inner_validation_weeks": inner_validation_weeks,
        "evaluation_years": list(range(first_evaluation_year, last_evaluation_year + 1)),
        "parameter_grid": spec.param_grid,
        "outer_folds": fold_records,
        "overall_out_of_sample_metrics": overall_metrics,
        "mean_fold_metrics": mean_fold_metrics,
        "calibration": calibration_summary,
        "leakage_controls": [
            "Every feature uses only strictly earlier tournament weeks.",
            "Each outer evaluation year is held out from both fitting and hyperparameter selection.",
            "Inner GridSearchCV uses recent fixed-duration validation windows and expands strictly through earlier whole tournament weeks.",
            "Imputation, scaling, categorical encoding, and model fitting occur inside each inner training split.",
            "No current-match score, duration, serve statistic, winner/loser field, or weather field is a model input.",
            "2026 is excluded from tuning and evaluation.",
        ],
    }
    return (
        artifact,
        report,
        pd.concat(all_grid_results, ignore_index=True),
        predictions,
        calibration,
    )


def write_model_outputs(
    *,
    artifact: dict[str, object],
    report: dict[str, object],
    grid_results: pd.DataFrame,
    predictions: pd.DataFrame,
    calibration: pd.DataFrame,
    spec: ModelSpec,
    models_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    """Write new tuning artifacts without touching the step-06/07 outputs."""
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": models_dir / f"{spec.name}_nested_walk_forward_latest_fold.joblib",
        "report": reports_dir / f"{spec.name}_nested_walk_forward.json",
        "grid_results": reports_dir / f"{spec.name}_inner_grid_results.csv",
        "predictions": reports_dir / f"{spec.name}_outer_fold_predictions.csv",
        "calibration": reports_dir / f"{spec.name}_calibration.csv",
        "reliability_plot": reports_dir / f"{spec.name}_reliability.html",
    }
    joblib.dump(artifact, paths["model"])
    paths["report"].write_text(
        json.dumps(report, indent=2, default=json_default), encoding="utf-8"
    )
    grid_results.to_csv(paths["grid_results"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    calibration.to_csv(paths["calibration"], index=False)
    reliability_figure(calibration, spec.name).write_html(
        paths["reliability_plot"], include_plotlyjs="cdn", full_html=True
    )
    return paths


def parse_args() -> argparse.Namespace:
    """Define the small, explicit tuning command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=sorted(MODEL_SPECS),
        help="Models to tune (default: all available models).",
    )
    parser.add_argument(
        "--first-evaluation-year", type=int, default=DEFAULT_FIRST_EVALUATION_YEAR
    )
    parser.add_argument(
        "--last-evaluation-year", type=int, default=DEFAULT_LAST_EVALUATION_YEAR
    )
    parser.add_argument(
        "--train-window-years", type=int, default=DEFAULT_TRAIN_WINDOW_YEARS
    )
    parser.add_argument("--inner-splits", type=int, default=DEFAULT_INNER_SPLITS)
    parser.add_argument(
        "--inner-validation-weeks",
        type=int,
        default=DEFAULT_INNER_VALIDATION_WEEKS,
        help=(
            "Duration of each recent fixed inner validation window in weeks "
            f"(default: {DEFAULT_INNER_VALIDATION_WEEKS})."
        ),
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=DEFAULT_CALIBRATION_BINS,
        help=(
            "Number of fixed probability bins for the out-of-fold reliability "
            f"report (default: {DEFAULT_CALIBRATION_BINS})."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="GridSearchCV workers; use 1 to limit CPU use (default: -1).",
    )
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(set(args.models)) != len(args.models):
        raise ValueError("Pass each model name at most once to --models.")
    features = load_features(args.input)
    comparison_rows: list[dict[str, object]] = []

    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        artifact, report, grid_results, predictions, calibration = nested_walk_forward_tuning(
            features,
            spec,
            first_evaluation_year=args.first_evaluation_year,
            last_evaluation_year=args.last_evaluation_year,
            train_window_years=args.train_window_years,
            inner_splits=args.inner_splits,
            inner_validation_weeks=args.inner_validation_weeks,
            calibration_bins=args.calibration_bins,
            n_jobs=args.n_jobs,
        )
        paths = write_model_outputs(
            artifact=artifact,
            report=report,
            grid_results=grid_results,
            predictions=predictions,
            calibration=calibration,
            spec=spec,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
        )
        print(f"Wrote {model_name} model artifact to {paths['model']}")
        print(f"Wrote {model_name} tuning report to {paths['report']}")
        print(f"Wrote {model_name} grid results to {paths['grid_results']}")
        print(f"Wrote {model_name} out-of-fold predictions to {paths['predictions']}")
        print(f"Wrote {model_name} calibration table to {paths['calibration']}")
        print(f"Wrote {model_name} reliability plot to {paths['reliability_plot']}")
        comparison_rows.append(
            {
                "model": model_name,
                "feature_count": report["feature_count"],
                **report["overall_out_of_sample_metrics"],
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("log_loss")
    comparison_path = args.reports_dir / "nested_walk_forward_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print("\nNested walk-forward comparison:")
    print(comparison.to_string(index=False))
    print(f"Wrote comparison to {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
