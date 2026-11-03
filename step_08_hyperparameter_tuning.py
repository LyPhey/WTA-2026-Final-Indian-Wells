"""Nested chronological hyperparameter tuning for the WTA winner models.

For every outer evaluation year, this script trains on only the immediately
preceding calendar years.  Inside that outer training window, GridSearchCV uses
expanding, tournament-week-aware validation splits to select hyperparameters.
The outer year is never used to select a configuration.

The default grids are intentionally small.  Extend MODEL_SPECS with another
ModelSpec rather than adding a one-off tuning loop for each future model.

Run from the project directory:
    python step_08_hyperparameter_tuning.py

For a quicker smoke run of logistic regression only:
    python step_08_hyperparameter_tuning.py --models logistic_regression \\
        --first-evaluation-year 2025 --last-evaluation-year 2025 --inner-splits 2
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def expanding_week_splits(
    training: pd.DataFrame, inner_splits: int
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, str]]]:
    """Create expanding validation folds that never split a tournament week.

    The first contiguous date block seeds training.  Each later block is
    validation, while training expands to all earlier tournament weeks.  The
    returned positional indices are safe to pass directly to GridSearchCV.
    """
    if inner_splits < 2:
        raise ValueError("--inner-splits must be at least 2.")
    unique_dates = pd.Index(training["match_date"].drop_duplicates().sort_values())
    if len(unique_dates) < inner_splits + 1:
        raise ValueError(
            f"Need at least {inner_splits + 1} distinct tournament weeks for "
            f"{inner_splits} inner splits; found {len(unique_dates)}."
        )

    date_blocks = [
        block for block in np.array_split(unique_dates.to_numpy(), inner_splits + 1) if len(block)
    ]
    if len(date_blocks) != inner_splits + 1:
        raise ValueError("Unable to construct the requested non-empty inner time splits.")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    metadata: list[dict[str, str]] = []
    dates = training["match_date"]
    for split_number, validation_dates in enumerate(date_blocks[1:], start=1):
        validation_start = pd.Timestamp(validation_dates[0])
        validation_end = pd.Timestamp(validation_dates[-1])
        train_index = np.flatnonzero((dates < validation_start).to_numpy())
        validation_index = np.flatnonzero(
            dates.between(validation_start, validation_end).to_numpy()
        )
        if not len(train_index) or not len(validation_index):
            raise ValueError(f"Inner split {split_number} is unexpectedly empty.")
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
            }
        )
    return splits, metadata


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
    n_jobs: int,
) -> tuple[dict[str, object], dict[str, object], pd.DataFrame]:
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
        inner_cv, inner_windows = expanding_week_splits(training, inner_splits)
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
        all_predictions.append(
            pd.DataFrame(
                {
                    "evaluation_year": evaluation_year,
                    "match_date": evaluation["match_date"],
                    "actual_player_a_won": evaluation[TARGET_COLUMN],
                    "probability_player_a_wins": probability,
                }
            )
        )
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
        "evaluation_years": list(range(first_evaluation_year, last_evaluation_year + 1)),
        "parameter_grid": spec.param_grid,
        "outer_folds": fold_records,
        "overall_out_of_sample_metrics": overall_metrics,
        "mean_fold_metrics": mean_fold_metrics,
        "leakage_controls": [
            "Every feature uses only strictly earlier tournament weeks.",
            "Each outer evaluation year is held out from both fitting and hyperparameter selection.",
            "Inner GridSearchCV folds expand strictly through earlier whole tournament weeks.",
            "Imputation, scaling, categorical encoding, and model fitting occur inside each inner training split.",
            "No current-match score, duration, serve statistic, winner/loser field, or weather field is a model input.",
            "2026 is excluded from tuning and evaluation.",
        ],
    }
    return artifact, report, pd.concat(all_grid_results, ignore_index=True)


def write_model_outputs(
    *,
    artifact: dict[str, object],
    report: dict[str, object],
    grid_results: pd.DataFrame,
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
    }
    joblib.dump(artifact, paths["model"])
    paths["report"].write_text(
        json.dumps(report, indent=2, default=json_default), encoding="utf-8"
    )
    grid_results.to_csv(paths["grid_results"], index=False)
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
        help="Models to tune (default: logistic_regression random_forest).",
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
        artifact, report, grid_results = nested_walk_forward_tuning(
            features,
            spec,
            first_evaluation_year=args.first_evaluation_year,
            last_evaluation_year=args.last_evaluation_year,
            train_window_years=args.train_window_years,
            inner_splits=args.inner_splits,
            n_jobs=args.n_jobs,
        )
        paths = write_model_outputs(
            artifact=artifact,
            report=report,
            grid_results=grid_results,
            spec=spec,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
        )
        print(f"Wrote {model_name} model artifact to {paths['model']}")
        print(f"Wrote {model_name} tuning report to {paths['report']}")
        print(f"Wrote {model_name} grid results to {paths['grid_results']}")
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
