"""Walk-forward backtest for the gradient-boosting candidate model.

This candidate uses the same rolling annual windows as step 06 so its results
can be compared fairly with the logistic-regression baseline. The default
configuration refits once for each evaluation year from 2020 through 2025 and
keeps 2026 entirely outside model fitting and selection.

Run from the project directory:
    python step_07_train_candidate_models.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from collections.abc import Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


INPUT_PATH = Path("data/processed/wta_pre_match_features.csv")
TARGET_COLUMN = "target_player_a_won"
MODEL_NAME = "hist_gradient_boosting_v1"
METADATA_COLUMNS = {
    "feature_version",
    "match_date",
    "feature_cutoff_date",
    "tourney_name",
    "player_a",
    "player_b",
    TARGET_COLUMN,
}
FORBIDDEN_FEATURE_PREFIXES = ("w_", "l_", "winner_", "loser_", "weather_")
FORBIDDEN_FEATURE_COLUMNS = {
    "score",
    "minutes",
    "source_record_count",
    "source_providers",
    "retained_provider",
}
CATEGORICAL_COLUMNS = ["surface", "tourney_level", "round"]


def candidate_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return permitted pre-match inputs and reject leakage-prone columns."""
    forbidden = sorted(
        column
        for column in features.columns
        if column.startswith(FORBIDDEN_FEATURE_PREFIXES)
        or column in FORBIDDEN_FEATURE_COLUMNS
    )
    if forbidden:
        raise ValueError(f"Feature data contains leakage-prone columns: {forbidden}")
    columns = [column for column in features.columns if column not in METADATA_COLUMNS]
    missing_categories = sorted(set(CATEGORICAL_COLUMNS) - set(columns))
    if missing_categories:
        raise ValueError(f"Feature data is missing categories: {missing_categories}")
    if not columns:
        raise ValueError("No candidate-model features remain after metadata exclusion.")
    return columns


def load_pre_match_features(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load and validate the feature table used by each walk-forward fold."""
    features = pd.read_csv(path, low_memory=False)
    required = {TARGET_COLUMN, "match_date", "feature_version"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")
    if features["feature_version"].nunique(dropna=False) != 1:
        raise ValueError("Input data mixes feature-building versions.")
    if not features[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("The target must be a binary Player A outcome.")
    features = features.copy()
    features["match_date"] = pd.to_datetime(features["match_date"], errors="coerce")
    if features["match_date"].isna().any():
        raise ValueError("Input data contains invalid match dates.")
    candidate_feature_columns(features)
    return features.sort_values("match_date", kind="stable").reset_index(drop=True)


def annual_fold(
    features: pd.DataFrame, evaluation_year: int, train_window_years: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one strict rolling-window training/evaluation partition."""
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


def build_candidate_pipeline(feature_columns: list[str]) -> Pipeline:
    """Build the user's gradient-boosting candidate for one training fold."""
    numeric_columns = [
        column for column in feature_columns if column not in CATEGORICAL_COLUMNS
    ]
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=15,
                    random_state=42,
                ),
            ),
        ]
    )

def build_random_forest_pipeline(feature_columns: list[str]) -> Pipeline:
    numeric_columns = [
        column 
        for column in feature_columns
        if column not in CATEGORICAL_COLUMNS
    ]

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator = True),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy = 'most_frequent')),
                        ('encoder', OneHotEncoder(handle_unknown='ignore')),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    n_jobs=-1,
                    random_state = 42,
                ),
            ),
        ]
    )


def calculate_metrics(
    actual: pd.Series, probability_player_a_wins: pd.Series
) -> dict[str, float]:
    """Calculate outcome and probability quality on held-out matches."""
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


def walk_forward_backtest(
    features: pd.DataFrame,
    first_evaluation_year: int,
    last_evaluation_year: int,
    train_window_years: int,
    *,
    model_name: str,
    pipeline_builder: Callable[[list[str]], Pipeline],
) -> tuple[dict[str, object], dict[str, object]]:
    """Refit the candidate per annual fold and return its audit artifacts."""
    if train_window_years < 1:
        raise ValueError("The training window must include at least one year.")
    if last_evaluation_year < first_evaluation_year:
        raise ValueError("The final evaluation year must not precede the first year.")
    if last_evaluation_year >= 2026:
        raise ValueError("Keep 2026 out of candidate-model selection and evaluation.")

    feature_columns = candidate_feature_columns(features)
    fold_records: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    latest_pipeline: Pipeline | None = None
    latest_training: pd.DataFrame | None = None

    for evaluation_year in range(first_evaluation_year, last_evaluation_year + 1):
        training, evaluation = annual_fold(
            features, evaluation_year, train_window_years
        )
        pipeline = pipeline_builder(feature_columns)

        pipeline.fit(training[feature_columns], training[TARGET_COLUMN])
        probability = pd.Series(
            pipeline.predict_proba(evaluation[feature_columns])[:, 1],
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
        "feature_columns": feature_columns,
        "feature_version": features["feature_version"].iloc[0],
        "purpose": "latest walk-forward candidate fold; not a match-day production model",
        "training_window_years": train_window_years,
        "evaluation_year": last_evaluation_year,
        "training_start_date": latest_training["match_date"].min().date().isoformat(),
        "training_end_date": latest_training["match_date"].max().date().isoformat(),
    }
    report = {
        "model": model_name,
        "backtest_method": "annual rolling-window walk-forward",
        "feature_version": artifact["feature_version"],
        "training_window_years": train_window_years,
        "evaluation_years": list(
            range(first_evaluation_year, last_evaluation_year + 1)
        ),
        "feature_count": len(feature_columns),
        "folds": fold_records,
        "overall_out_of_sample_metrics": overall_metrics,
        "mean_fold_metrics": mean_fold_metrics,
        "leakage_controls": [
            "Every historical feature uses only strictly earlier tournament weeks.",
            "Each fold refits preprocessing and the model using a strictly earlier rolling window.",
            "No current-match score, duration, serve statistic, winner/loser field, or weather field is a model input.",
            "2026 is excluded from fitting, model selection, and evaluation.",
        ],
    }
    return artifact, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--first-evaluation-year", type=int, default=2020)
    parser.add_argument("--last-evaluation-year", type=int, default=2025)
    parser.add_argument("--train-window-years", type=int, default=4)
    args = parser.parse_args()

    features = load_pre_match_features(args.input)

    candidates = [
        ("hist_gradient_boosting_v1", build_candidate_pipeline),
        ("random_forest_v1", build_random_forest_pipeline),
    ]

    comparison_rows = []

    for model_name, pipeline_builder in candidates:
        artifact, report = walk_forward_backtest(
            features,
            args.first_evaluation_year,
            args.last_evaluation_year,
            args.train_window_years,
            model_name=model_name,
            pipeline_builder=pipeline_builder,
        )

        model_path = Path(f"models/{model_name}_walk_forward_latest_fold.joblib")
        report_path = Path(f"reports/{model_name}_walk_forward.json")

        model_path.parent.mkdir(exist_ok=True)
        report_path.parent.mkdir(exist_ok=True)

        joblib.dump(artifact, model_path)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        comparison_rows.append(
            {
                "model": model_name,
                "feature_count": report["feature_count"],
                **report["overall_out_of_sample_metrics"],
            }
        )
    comparison = pd.DataFrame(comparison_rows).sort_values(
        "log_loss",
        ascending=True,
    )

    comparison.to_csv(
        "reports/candidate_model_comparison.csv",
        index=False,
    )

    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
