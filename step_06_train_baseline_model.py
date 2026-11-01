"""Walk-forward backtest of a rolling-window logistic-regression baseline.

For each evaluation year, this script refits the full preprocessing and model
using only the preceding calendar years. The default six folds use four years
of training data to evaluate 2020 through 2025. No 2026 result is used for
model fitting, model selection, or baseline evaluation.

Run from the project directory:
    python step_06_train_baseline_model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_PATH = Path("data/processed/wta_pre_match_features.csv")
MODEL_PATH = Path("models/wta_logistic_regression_walk_forward_baseline.joblib")
REPORT_PATH = Path("reports/wta_logistic_regression_walk_forward_baseline.json")
TARGET_COLUMN = "target_player_a_won"
ROLLING_WINDOWS_WEEKS = (13, 26, 52)
ROLLING_MODEL_FIELDS = [
    "matches",
    "win_rate",
    "surface_matches",
    "surface_win_rate",
    "aces_per_match",
    "double_faults_per_match",
    "first_serve_won_pct",
    "second_serve_won_pct",
    "serve_points_won_pct",
    "match_minutes",
]
NUMERIC_FEATURES = [
    "rank_advantage_player_a",
    "rank_points_advantage_player_a",
    "elo_advantage_player_a",
    "surface_elo_advantage_player_a",
    "weeks_since_last_match_advantage_player_a",
    "recent_10_matches_advantage_player_a",
    "recent_10_win_rate_advantage_player_a",
    "h2h_prior_matches",
    "h2h_player_a_win_rate",
    "surface_h2h_prior_matches",
    "surface_h2h_player_a_win_rate",
    *[
        f"prior_{weeks}w_{field}_advantage_player_a"
        for weeks in ROLLING_WINDOWS_WEEKS
        for field in ROLLING_MODEL_FIELDS
    ],
]
CATEGORICAL_FEATURES = ["surface", "tourney_level", "round"]
MODEL_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def build_pipeline() -> Pipeline:
    """Create a new pipeline that learns preprocessing only from one fold."""
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
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2_000, random_state=42)),
        ]
    )


def validate_input(features: pd.DataFrame) -> pd.DataFrame:
    """Validate the leakage-safe feature table before creating any fold."""
    required = {TARGET_COLUMN, "match_date", "feature_version", *MODEL_FEATURES}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Training input is missing required columns: {missing}")
    if features["feature_version"].nunique(dropna=False) != 1:
        raise ValueError("Training input mixes feature-building versions.")
    if not features[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Target must be a binary Player A outcome.")
    prepared = features.copy()
    prepared["match_date"] = pd.to_datetime(prepared["match_date"], errors="coerce")
    if prepared["match_date"].isna().any():
        raise ValueError("Training input contains invalid match dates.")
    return prepared


def calculate_metrics(
    actual: pd.Series, probability_player_a_wins: pd.Series
) -> dict[str, float]:
    """Calculate classification and probability metrics from held-out rows."""
    predicted = (probability_player_a_wins >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(actual, predicted)), 6),
        "brier_score": round(
            float(brier_score_loss(actual, probability_player_a_wins)), 6
        ),
        "log_loss": round(
            float(log_loss(actual, probability_player_a_wins, labels=[0, 1])), 6
        ),
        "roc_auc": round(float(roc_auc_score(actual, probability_player_a_wins)), 6),
    }


def annual_fold(
    features: pd.DataFrame, evaluation_year: int, train_window_years: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one rolling history window and its strictly later evaluation year."""
    evaluation_start = pd.Timestamp(year=evaluation_year, month=1, day=1)
    evaluation_end = pd.Timestamp(year=evaluation_year, month=12, day=31)
    training_start = evaluation_start - pd.DateOffset(years=train_window_years)
    training_end = evaluation_start - pd.Timedelta(days=1)
    training = features.loc[
        features["match_date"].between(training_start, training_end)
    ].copy()
    evaluation = features.loc[
        features["match_date"].between(evaluation_start, evaluation_end)
    ].copy()
    if training.empty or evaluation.empty:
        raise ValueError(f"Fold {evaluation_year} has an empty training or evaluation set.")
    if training["match_date"].max() >= evaluation["match_date"].min():
        raise ValueError(f"Fold {evaluation_year} is not strictly time ordered.")
    if training[TARGET_COLUMN].nunique() != 2:
        raise ValueError(f"Fold {evaluation_year} training data lacks an outcome class.")
    return training, evaluation


def walk_forward_backtest(
    features: pd.DataFrame,
    first_evaluation_year: int,
    last_evaluation_year: int,
    train_window_years: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Refit and score one rolling model per annual evaluation fold."""
    if train_window_years < 1:
        raise ValueError("The training window must include at least one year.")
    if last_evaluation_year < first_evaluation_year:
        raise ValueError("The final evaluation year must not precede the first year.")

    prepared = validate_input(features)
    if last_evaluation_year >= 2026:
        raise ValueError("Keep 2026 out of baseline model selection and evaluation.")

    fold_records: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    latest_pipeline: Pipeline | None = None
    latest_training: pd.DataFrame | None = None

    for evaluation_year in range(first_evaluation_year, last_evaluation_year + 1):
        training, evaluation = annual_fold(
            prepared, evaluation_year, train_window_years
        )
        pipeline = build_pipeline()
        pipeline.fit(training[MODEL_FEATURES], training[TARGET_COLUMN])
        probability = pd.Series(
            pipeline.predict_proba(evaluation[MODEL_FEATURES])[:, 1],
            index=evaluation.index,
            name="probability_player_a_wins",
        )
        metrics = calculate_metrics(evaluation[TARGET_COLUMN], probability)
        fold_records.append(
            {
                "evaluation_year": evaluation_year,
                "training_rows": int(len(training)),
                "training_start_date": training["match_date"].min().date().isoformat(),
                "training_end_date": training["match_date"].max().date().isoformat(),
                "evaluation_rows": int(len(evaluation)),
                "evaluation_start_date": evaluation["match_date"].min().date().isoformat(),
                "evaluation_end_date": evaluation["match_date"].max().date().isoformat(),
                **metrics,
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "evaluation_year": evaluation_year,
                    "match_date": evaluation["match_date"],
                    "actual_player_a_won": evaluation[TARGET_COLUMN],
                    "probability_player_a_wins": probability,
                }
            )
        )
        latest_pipeline = pipeline
        latest_training = training

    if latest_pipeline is None or latest_training is None:
        raise ValueError("No walk-forward folds were created.")

    all_predictions = pd.concat(predictions, ignore_index=True)
    overall_metrics = calculate_metrics(
        all_predictions["actual_player_a_won"],
        all_predictions["probability_player_a_wins"],
    )
    fold_metrics = pd.DataFrame(fold_records)
    mean_fold_metrics = {
        metric: round(float(fold_metrics[metric].mean()), 6)
        for metric in overall_metrics
    }
    artifact = {
        "pipeline": latest_pipeline,
        "model_features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_version": prepared["feature_version"].iloc[0],
        "purpose": "latest walk-forward backtest fold; not a match-day production model",
        "training_window_years": train_window_years,
        "evaluation_year": last_evaluation_year,
        "training_start_date": latest_training["match_date"].min().date().isoformat(),
        "training_end_date": latest_training["match_date"].max().date().isoformat(),
    }
    report = {
        "model": "logistic_regression",
        "backtest_method": "annual rolling-window walk-forward",
        "feature_version": artifact["feature_version"],
        "training_window_years": train_window_years,
        "evaluation_years": list(
            range(first_evaluation_year, last_evaluation_year + 1)
        ),
        "folds": fold_records,
        "overall_out_of_sample_metrics": overall_metrics,
        "mean_fold_metrics": mean_fold_metrics,
        "leakage_controls": [
            "Every historical feature uses only strictly earlier tournament weeks.",
            "Each fold uses training rows from a strictly earlier rolling window.",
            "Imputation, scaling, and categorical encoding are refit inside every fold.",
            "No current-match score, duration, serve statistic, winner/loser field, or weather field is a model input.",
            "2026 is excluded from fitting, model selection, and evaluation.",
        ],
    }
    return artifact, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--report-output", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--first-evaluation-year",
        type=int,
        default=2020,
        help="First annual out-of-sample evaluation fold (default: 2020).",
    )
    parser.add_argument(
        "--last-evaluation-year",
        type=int,
        default=2025,
        help="Last annual out-of-sample evaluation fold (default: 2025).",
    )
    parser.add_argument(
        "--train-window-years",
        type=int,
        default=4,
        help="Number of immediately preceding calendar years used per fold.",
    )
    args = parser.parse_args()

    features = pd.read_csv(args.input, low_memory=False)
    artifact, report = walk_forward_backtest(
        features,
        args.first_evaluation_year,
        args.last_evaluation_year,
        args.train_window_years,
    )
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_output)
    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote latest-fold baseline model to {args.model_output}")
    print(f"Wrote walk-forward report to {args.report_output}")
    print("Annual held-out metrics:")
    for fold in report["folds"]:
        print(
            f"  {fold['evaluation_year']}: "
            f"log loss {fold['log_loss']:.3f}, "
            f"Brier {fold['brier_score']:.3f}, "
            f"ROC AUC {fold['roc_auc']:.3f}"
        )
    print("Overall out-of-sample metrics:")
    for metric, value in report["overall_out_of_sample_metrics"].items():
        print(f"  {metric}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
