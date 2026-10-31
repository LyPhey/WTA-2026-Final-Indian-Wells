"""Load and split the leakage-safe dataset for future candidate models.

This step deliberately defines no machine-learning models. It provides one
reusable, chronological train/evaluation split so later candidate-model code
does not accidentally use random sampling or include 2026 data by default.

Run from the project directory:
    python step_07_train_candidate_models.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


INPUT_PATH = Path("data/processed/wta_pre_match_features.csv")
TARGET_COLUMN = "target_player_a_won"
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


@dataclass
class TemporalDatasetSplit:
    """Leakage-safe data partitions available to a future candidate model."""

    training: pd.DataFrame
    evaluation: pd.DataFrame
    future: pd.DataFrame
    feature_columns: list[str]

    @property
    def x_train(self) -> pd.DataFrame:
        """Return model inputs for fitting once a candidate is selected."""
        return self.training[self.feature_columns].copy()

    @property
    def y_train(self) -> pd.Series:
        """Return binary Player A outcome labels for fitting."""
        return self.training[TARGET_COLUMN].copy()

    @property
    def x_evaluation(self) -> pd.DataFrame:
        """Return held-out model inputs for one later-period evaluation."""
        return self.evaluation[self.feature_columns].copy()

    @property
    def y_evaluation(self) -> pd.Series:
        """Return held-out binary Player A outcome labels."""
        return self.evaluation[TARGET_COLUMN].copy()


def candidate_feature_columns(features: pd.DataFrame) -> list[str]:
    """Return permitted pre-match inputs while rejecting leakage-prone fields."""
    forbidden = sorted(
        column
        for column in features.columns
        if column.startswith(FORBIDDEN_FEATURE_PREFIXES)
        or column in FORBIDDEN_FEATURE_COLUMNS
    )
    if forbidden:
        raise ValueError(f"Feature data contains leakage-prone columns: {forbidden}")
    columns = [column for column in features.columns if column not in METADATA_COLUMNS]
    if not columns:
        raise ValueError("No candidate-model features remain after metadata exclusion.")
    return columns


def load_pre_match_features(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load and validate the pre-match dataset without fitting any model."""
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


def build_temporal_split(
    features: pd.DataFrame,
    train_through: pd.Timestamp,
    evaluation_year: int,
) -> TemporalDatasetSplit:
    """Make a strict past-to-future split and leave later rows untouched."""
    if evaluation_year <= train_through.year:
        raise ValueError("The evaluation year must be later than the training cutoff year.")

    feature_columns = candidate_feature_columns(features)
    training = features.loc[features["match_date"].le(train_through)].copy()
    evaluation = features.loc[
        features["match_date"].dt.year.eq(evaluation_year)
    ].copy()
    future = features.loc[
        features["match_date"].gt(train_through)
        & ~features["match_date"].dt.year.eq(evaluation_year)
    ].copy()

    if training.empty or evaluation.empty:
        raise ValueError("The requested temporal split has an empty partition.")
    if training["match_date"].max() >= evaluation["match_date"].min():
        raise ValueError("Evaluation matches must occur strictly after training matches.")
    if training[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Training data must contain both outcome classes.")

    return TemporalDatasetSplit(
        training=training,
        evaluation=evaluation,
        future=future,
        feature_columns=feature_columns,
    )


def print_split_summary(split: TemporalDatasetSplit) -> None:
    """Print an audit-friendly split summary without defining a candidate model."""
    print(f"Candidate-model feature columns: {len(split.feature_columns):,}")
    print(
        "Training rows: "
        f"{len(split.training):,} "
        f"({split.training['match_date'].min().date()} to "
        f"{split.training['match_date'].max().date()})"
    )
    print(
        "Held-out evaluation rows: "
        f"{len(split.evaluation):,} "
        f"({split.evaluation['match_date'].min().date()} to "
        f"{split.evaluation['match_date'].max().date()})"
    )
    print(f"Future rows left out of this run: {len(split.future):,}")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument(
        "--train-through",
        default="2024-12-31",
        help="Inclusive last date for candidate-model fitting data.",
    )
    parser.add_argument(
        "--evaluation-year",
        type=int,
        default=2025,
        help="Later year held out for candidate-model evaluation.",
    )
    args = parser.parse_args()

    features = load_pre_match_features(args.input)
    split = build_temporal_split(
        features, pd.Timestamp(args.train_through), args.evaluation_year
    )

    print_split_summary(split)

    categorical_columns = ["surface", "tourney_level", "round"]
    numeric_columns = [
        column 
        for column in split.feature_columns
        if column not in categorical_columns
    ]

    preprocessor = ColumnTransformer(
        transformers= [
            (
                "numeric",
                SimpleImputer(strategy = "median", add_indicator = True),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps = [
                        ("imputer", SimpleImputer(strategy = 'most_frequent')),
                        (
                            'encoder',
                            OrdinalEncoder(
                                handle_unknown = 'use_encoded_value',
                                unknown_value = -1,
                            ),
                        ),
                    ]
                ),
                categorical_columns,

            ),
        ]
    )

    candidate_model = Pipeline(
        steps = [
        ('preprocessor', preprocessor),
        (
            'model',
            HistGradientBoostingClassifier(
                learning_rate = 0.05,
                max_iter=300,
                max_leaf_nodes=15,
                random_state=42
            )

        )
        ]
    )

    candidate_model.fit(split.x_train, split.y_train)

    probability_player_a_wins = candidate_model.predict_proba(
        split.x_evaluation
    )[:, 1]

    predicted_player_a_wins = (probability_player_a_wins >= 0.5).astype(int)

    print("\nCandidate-model results on held-out 2025 data")
    print(
        f"Accuracy: "
        f"{accuracy_score(split.y_evaluation, predicted_player_a_wins):.3f}"
    )

    print(
        f"Brier score: "
        f"{brier_score_loss(split.y_evaluation, probability_player_a_wins):.3f}"
    )

    print(
        f"Log loss: "
        f"{log_loss(split.y_evaluation, probability_player_a_wins):.3f}"
    )

    print(
        f"ROC AUC: "
        f"{roc_auc_score(split.y_evaluation, probability_player_a_wins):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
