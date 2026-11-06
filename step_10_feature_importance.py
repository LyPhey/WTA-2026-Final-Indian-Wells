"""Measure held-out feature importance for the selected Step 08 finalist.

The script refits the model chosen by Step 08 on each historical outer training
window, then measures the increase in held-out log loss after permuting one
feature or one related feature group. It never removes features or overwrites a
model artifact; use its results to design a separate ablation experiment.

Run from the project directory:
    python step_10_feature_importance.py

For a quick single-year smoke run:
    python step_10_feature_importance.py --first-evaluation-year 2025 \
        --last-evaluation-year 2025 --n-repeats 2
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

from step_06_train_baseline_model import CATEGORICAL_FEATURES, MODEL_FEATURES, TARGET_COLUMN
from step_08_hyperparameter_tuning import MODEL_SPECS, annual_outer_fold, load_features


DEFAULT_REPORTS_DIR = Path("reports/hyperparameter_tuning")
DEFAULT_OUTPUT_DIR = Path("reports/feature_importance")
DEFAULT_REPEATS = 20
DEFAULT_N_JOBS = 1


def load_tuning_reports(reports_dir: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    """Load individual Step 08 reports that can be reproduced with this project."""
    reports: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in reports_dir.glob("*_nested_walk_forward.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        model_name = report.get("model")
        metrics = report.get("overall_out_of_sample_metrics", {})
        if model_name in MODEL_SPECS and "log_loss" in metrics:
            reports[str(model_name)] = (report, path)
    if not reports:
        raise FileNotFoundError(
            f"No compatible Step 08 reports were found in {reports_dir}. "
            "Run step_08_hyperparameter_tuning.py first."
        )
    return reports


def select_model_report(
    reports_dir: Path, requested_model: str
) -> tuple[str, dict[str, Any], Path]:
    """Choose a requested finalist or the available finalist with lowest log loss."""
    reports = load_tuning_reports(reports_dir)
    if requested_model != "auto":
        if requested_model not in reports:
            raise ValueError(
                f"No compatible Step 08 report exists for {requested_model!r}. "
                f"Available: {', '.join(sorted(reports))}."
            )
        report, path = reports[requested_model]
        return requested_model, report, path
    model_name, (report, path) = min(
        reports.items(),
        key=lambda item: float(item[1][0]["overall_out_of_sample_metrics"]["log_loss"]),
    )
    return model_name, report, path


def feature_groups(features: list[str]) -> dict[str, list[str]]:
    """Keep correlated pre-match signals together for grouped permutations."""
    groups: dict[str, list[str]] = {
        "Official rankings": [],
        "Elo ratings": [],
        "Recent form": [],
        "Head-to-head": [],
        "13-week rolling form": [],
        "26-week rolling form": [],
        "52-week rolling form": [],
        "Tournament context": [],
        "Other": [],
    }
    for feature in features:
        if feature in CATEGORICAL_FEATURES:
            group = "Tournament context"
        elif feature.startswith(("rank_advantage", "rank_points_advantage")):
            group = "Official rankings"
        elif feature in {"elo_advantage_player_a", "surface_elo_advantage_player_a"}:
            group = "Elo ratings"
        elif "h2h" in feature:
            group = "Head-to-head"
        elif feature.startswith("prior_13w_"):
            group = "13-week rolling form"
        elif feature.startswith("prior_26w_"):
            group = "26-week rolling form"
        elif feature.startswith("prior_52w_"):
            group = "52-week rolling form"
        elif feature.startswith(("weeks_since_last_match", "recent_10_")):
            group = "Recent form"
        else:
            group = "Other"
        groups[group].append(feature)
    return {group: columns for group, columns in groups.items() if columns}


def score_log_loss(pipeline: Any, features: pd.DataFrame, outcome: pd.Series) -> float:
    """Return positive log loss from a fitted pipeline on untouched rows."""
    probability = pipeline.predict_proba(features)[:, 1]
    return float(log_loss(outcome, probability, labels=[0, 1]))


def grouped_permutation_importance(
    pipeline: Any,
    features: pd.DataFrame,
    outcome: pd.Series,
    groups: dict[str, list[str]],
    *,
    baseline_log_loss: float,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """Jointly permute related columns to avoid splitting correlated signal."""
    generator = np.random.default_rng(random_state)
    records: list[dict[str, float | int | str]] = []
    for group_name, columns in groups.items():
        increases: list[float] = []
        for _ in range(n_repeats):
            permuted = features.copy()
            order = generator.permutation(len(permuted))
            permuted.loc[:, columns] = permuted.iloc[order][columns].to_numpy()
            increases.append(score_log_loss(pipeline, permuted, outcome) - baseline_log_loss)
        records.append(
            {
                "feature_group": group_name,
                "feature_count": len(columns),
                "importance_mean_log_loss_increase": float(np.mean(increases)),
                "importance_std_across_repeats": float(np.std(increases, ddof=0)),
            }
        )
    return pd.DataFrame(records)


def summarize_importance(records: pd.DataFrame, name_column: str) -> pd.DataFrame:
    """Aggregate outer-fold importances and show their consistency over time."""
    summary = (
        records.groupby(name_column, as_index=False)
        .agg(
            mean_log_loss_increase=("importance_mean_log_loss_increase", "mean"),
            std_across_outer_folds=("importance_mean_log_loss_increase", "std"),
            positive_outer_fold_fraction=(
                "importance_mean_log_loss_increase",
                lambda values: float((values > 0).mean()),
            ),
            outer_fold_count=("evaluation_year", "nunique"),
        )
        .sort_values("mean_log_loss_increase", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    summary["std_across_outer_folds"] = summary["std_across_outer_folds"].fillna(0.0)
    return summary


def feature_importance_figure(
    feature_summary: pd.DataFrame, group_summary: pd.DataFrame, model_name: str
) -> go.Figure:
    """Build a decision-focused chart for individual and grouped importance."""
    top_features = feature_summary.head(25).sort_values(
        "mean_log_loss_increase", ascending=True
    )
    groups = group_summary.sort_values("mean_log_loss_increase", ascending=True)
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Top individual features", "Grouped feature families"],
        horizontal_spacing=0.16,
    )
    for column, frame, label in (
        (1, top_features, "feature"),
        (2, groups, "feature_group"),
    ):
        colors = np.where(
            frame["mean_log_loss_increase"].ge(0), "#2563eb", "#dc2626"
        )
        figure.add_trace(
            go.Bar(
                x=frame["mean_log_loss_increase"],
                y=frame[label],
                orientation="h",
                marker_color=colors,
                error_x={
                    "type": "data",
                    "array": frame["std_across_outer_folds"],
                    "visible": True,
                },
                customdata=frame[["positive_outer_fold_fraction", "outer_fold_count"]].to_numpy(),
                hovertemplate=(
                    f"{label.replace('_', ' ').title()}: %{{y}}"
                    "<br>Mean log-loss increase: %{x:.4f}"
                    "<br>Positive-fold fraction: %{customdata[0]:.0%}"
                    "<br>Outer folds: %{customdata[1]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_xaxes(
            title_text="Held-out log-loss increase after permutation", row=1, col=column
        )
    figure.update_layout(
        title=(
            f"{model_name}: held-out permutation importance "
            "(positive values are helpful)"
        ),
        template="plotly_white",
        height=max(560, 24 * len(top_features)),
        width=1_400,
    )
    return figure


def parse_args() -> argparse.Namespace:
    """Define explicit controls for a reproducible importance analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/wta_pre_match_features.csv"))
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        default="auto",
        help="Step 08 model to inspect, or 'auto' for the lowest-log-loss finalist.",
    )
    parser.add_argument("--n-repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--first-evaluation-year", type=int)
    parser.add_argument("--last-evaluation-year", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_repeats < 1:
        raise ValueError("--n-repeats must be at least 1.")
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero; use -1 for all available cores.")
    if (
        args.first_evaluation_year is not None
        and args.last_evaluation_year is not None
        and args.last_evaluation_year < args.first_evaluation_year
    ):
        raise ValueError("The final evaluation year must not precede the first year.")

    model_name, report, report_path = select_model_report(args.reports_dir, args.model)
    training_years = int(report["training_window_years"])
    outer_folds = report.get("outer_folds", [])
    if not outer_folds:
        raise ValueError(f"The Step 08 report for {model_name} has no outer-fold records.")
    selected_folds = [
        fold
        for fold in outer_folds
        if (
            args.first_evaluation_year is None
            or int(fold["evaluation_year"]) >= args.first_evaluation_year
        )
        and (
            args.last_evaluation_year is None
            or int(fold["evaluation_year"]) <= args.last_evaluation_year
        )
    ]
    if not selected_folds:
        raise ValueError("No Step 08 outer folds match the requested evaluation-year range.")

    features = load_features(args.input)
    groups = feature_groups(MODEL_FEATURES)
    individual_records: list[pd.DataFrame] = []
    group_records: list[pd.DataFrame] = []
    for fold in sorted(selected_folds, key=lambda item: int(item["evaluation_year"])):
        evaluation_year = int(fold["evaluation_year"])
        parameters = fold.get("best_params")
        if not isinstance(parameters, dict):
            raise ValueError(
                f"Step 08 report for {model_name}, outer {evaluation_year}, lacks best_params."
            )
        training, evaluation = annual_outer_fold(features, evaluation_year, training_years)
        pipeline = MODEL_SPECS[model_name].pipeline_builder().set_params(**parameters)
        pipeline.fit(training[MODEL_FEATURES], training[TARGET_COLUMN])
        evaluation_features = evaluation[MODEL_FEATURES]
        evaluation_outcome = evaluation[TARGET_COLUMN]
        baseline_log_loss = score_log_loss(pipeline, evaluation_features, evaluation_outcome)

        individual = permutation_importance(
            pipeline,
            evaluation_features,
            evaluation_outcome,
            scoring="neg_log_loss",
            n_repeats=args.n_repeats,
            random_state=42 + evaluation_year,
            n_jobs=args.n_jobs,
        )
        individual_records.append(
            pd.DataFrame(
                {
                    "evaluation_year": evaluation_year,
                    "baseline_log_loss": baseline_log_loss,
                    "feature": MODEL_FEATURES,
                    "importance_mean_log_loss_increase": individual.importances_mean,
                    "importance_std_across_repeats": individual.importances_std,
                }
            )
        )
        grouped = grouped_permutation_importance(
            pipeline,
            evaluation_features,
            evaluation_outcome,
            groups,
            baseline_log_loss=baseline_log_loss,
            n_repeats=args.n_repeats,
            random_state=10_000 + evaluation_year,
        )
        grouped.insert(0, "evaluation_year", evaluation_year)
        grouped.insert(1, "baseline_log_loss", baseline_log_loss)
        group_records.append(grouped)
        print(
            f"{model_name}, outer {evaluation_year}: baseline log loss "
            f"{baseline_log_loss:.3f}; computed {len(MODEL_FEATURES)} individual and "
            f"{len(groups)} grouped importances."
        )

    individual_results = pd.concat(individual_records, ignore_index=True)
    group_results = pd.concat(group_records, ignore_index=True)
    feature_summary = summarize_importance(individual_results, "feature")
    group_summary = summarize_importance(group_results, "feature_group")
    figure = feature_importance_figure(feature_summary, group_summary, model_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{model_name}_permutation_importance"
    paths = {
        "individual": args.output_dir / f"{prefix}_by_outer_fold.csv",
        "groups": args.output_dir / f"{prefix}_groups_by_outer_fold.csv",
        "feature_summary": args.output_dir / f"{prefix}_summary.csv",
        "group_summary": args.output_dir / f"{prefix}_group_summary.csv",
        "plot": args.output_dir / f"{prefix}.html",
        "metadata": args.output_dir / f"{prefix}_metadata.json",
    }
    individual_results.to_csv(paths["individual"], index=False)
    group_results.to_csv(paths["groups"], index=False)
    feature_summary.to_csv(paths["feature_summary"], index=False)
    group_summary.to_csv(paths["group_summary"], index=False)
    figure.write_html(paths["plot"], include_plotlyjs="cdn", full_html=True)
    paths["metadata"].write_text(
        json.dumps(
            {
                "model": model_name,
                "selection_rule": (
                    "requested explicitly"
                    if args.model != "auto"
                    else "lowest available Step 08 overall outer-fold log loss"
                ),
                "source_tuning_report": str(report_path),
                "report_modified_utc": datetime.fromtimestamp(
                    report_path.stat().st_mtime, timezone.utc
                ).isoformat(timespec="seconds"),
                "evaluation_years": [int(fold["evaluation_year"]) for fold in selected_folds],
                "training_window_years": training_years,
                "permutation_repeats": args.n_repeats,
                "scoring": "increase in held-out log loss after permutation",
                "automatic_feature_removal": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote individual outer-fold importance to {paths['individual']}")
    print(f"Wrote grouped outer-fold importance to {paths['groups']}")
    print(f"Wrote feature summary to {paths['feature_summary']}")
    print(f"Wrote grouped summary to {paths['group_summary']}")
    print(f"Wrote interactive importance plot to {paths['plot']}")
    print("No features were removed. Use ablation testing before excluding a feature group.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
