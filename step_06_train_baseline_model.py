"""Train and evaluate a leakage-safe logistic-regression winner baseline.

The default split trains on 2016-2024 and evaluates once on the later 2025
season. It deliberately excludes 2026, which remains available for future
match-day retraining after a prediction cutoff has been selected.

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
MODEL_PATH = Path("models/wta_logistic_regression_baseline.joblib")
REPORT_PATH = Path("reports/wta_logistic_regression_baseline.json")
TARGET_COLUMN = "target_player_a_won"
NUMERIC_FEATURES = [
    "rank_advantage_player_a",
    "rank_points_advantage_player_a",
    "prior_matches_advantage_player_a",
    "prior_win_rate_advantage_player_a",
    "prior_surface_matches_advantage_player_a",
    "prior_surface_win_rate_advantage_player_a",
    "recent_10_matches_advantage_player_a",
    "recent_10_win_rate_advantage_player_a",
    "prior_aces_per_match_advantage_player_a",
    "prior_double_faults_per_match_advantage_player_a",
    "prior_first_serve_won_pct_advantage_player_a",
    "prior_second_serve_won_pct_advantage_player_a",
    "prior_serve_points_won_pct_advantage_player_a",
    "prior_match_minutes_advantage_player_a",
    "h2h_prior_matches",
    "h2h_player_a_win_rate",
    "surface_h2h_prior_matches",
    "surface_h2h_player_a_win_rate",
]
CATEGORICAL_FEATURES = ["surface", "tourney_level", "round"]
MODEL_FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def build_pipeline() -> Pipeline:
    """Create one pipeline so preprocessing is fit only on the training period."""
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
    """Check the leakage-safe feature file before any model fit occurs."""
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


def train_and_evaluate(
    features: pd.DataFrame, train_through: pd.Timestamp, evaluation_year: int
) -> tuple[dict[str, object], dict[str, object]]:
    """Fit on a past period and evaluate once on a strictly later year."""
    prepared = validate_input(features)
    train = prepared.loc[prepared["match_date"].le(train_through)].copy()
    evaluation = prepared.loc[
        prepared["match_date"].dt.year.eq(evaluation_year)
    ].copy()
    if train.empty or evaluation.empty:
        raise ValueError("The requested temporal train/evaluation split has no rows.")
    if train["match_date"].max() >= evaluation["match_date"].min():
        raise ValueError("Evaluation matches must occur strictly after the training window.")
    if train[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Training data must contain both outcome classes.")

    pipeline = build_pipeline()
    pipeline.fit(train[MODEL_FEATURES], train[TARGET_COLUMN])
    probability_player_a_wins = pipeline.predict_proba(evaluation[MODEL_FEATURES])[:, 1]
    predicted_player_a_wins = (probability_player_a_wins >= 0.5).astype(int)

    metrics = {
        "accuracy": round(
            float(accuracy_score(evaluation[TARGET_COLUMN], predicted_player_a_wins)), 6
        ),
        "brier_score": round(
            float(brier_score_loss(evaluation[TARGET_COLUMN], probability_player_a_wins)),
            6,
        ),
        "log_loss": round(
            float(
                log_loss(
                    evaluation[TARGET_COLUMN], probability_player_a_wins, labels=[0, 1]
                )
            ),
            6,
        ),
        "roc_auc": round(
            float(roc_auc_score(evaluation[TARGET_COLUMN], probability_player_a_wins)),
            6,
        ),
    }
    artifact = {
        "pipeline": pipeline,
        "model_features": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_version": prepared["feature_version"].iloc[0],
        "train_through": train_through.date().isoformat(),
        "evaluation_year": evaluation_year,
    }
    report = {
        "model": "logistic_regression",
        "feature_version": artifact["feature_version"],
        "train_period": {
            "rows": int(len(train)),
            "first_match_date": train["match_date"].min().date().isoformat(),
            "last_match_date": train["match_date"].max().date().isoformat(),
        },
        "evaluation_period": {
            "rows": int(len(evaluation)),
            "first_match_date": evaluation["match_date"].min().date().isoformat(),
            "last_match_date": evaluation["match_date"].max().date().isoformat(),
        },
        "metrics": metrics,
        "leakage_controls": [
            "Every historical feature uses only strictly earlier tournament weeks.",
            "No current-match score, duration, serve statistic, winner/loser column, or weather field is a model input.",
            "Imputation, scaling, and categorical encoding are fit only on the training period.",
            "The 2025 evaluation season is held out from fitting; 2026 is excluded from this baseline run.",
        ],
    }
    return artifact, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--model-output", type=Path, default=MODEL_PATH)
    parser.add_argument("--report-output", type=Path, default=REPORT_PATH)
    parser.add_argument(
        "--train-through",
        default="2024-12-31",
        help="Inclusive last date for model fitting (default: 2024-12-31).",
    )
    parser.add_argument(
        "--evaluation-year",
        type=int,
        default=2025,
        help="Strictly later year used once for evaluation (default: 2025).",
    )
    args = parser.parse_args()

    train_through = pd.Timestamp(args.train_through)
    features = pd.read_csv(args.input, low_memory=False)
    artifact, report = train_and_evaluate(
        features, train_through, args.evaluation_year
    )
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_output)
    args.report_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote baseline model to {args.model_output}")
    print(f"Wrote evaluation report to {args.report_output}")
    print("Held-out metrics:")
    for metric, value in report["metrics"].items():
        print(f"  {metric}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
