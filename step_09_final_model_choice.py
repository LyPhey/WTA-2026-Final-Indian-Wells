"""Create a read-only visual report from Step 08 model-tuning outputs.

This script does not fit, tune, or overwrite a model. It reads the individual
Step 08 reports so running one model later does not hide the other candidates.

Run from the project directory:
    python step_09_final_model_choice.py

Open the generated report:
    reports/final_model_choice/final_model_choice.html
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_REPORTS_DIR = Path("reports/hyperparameter_tuning")
DEFAULT_OUTPUT_DIR = Path("reports/final_model_choice")
METRICS = (
    ("log_loss", "Log loss", "lower"),
    ("brier_score", "Brier score", "lower"),
    ("accuracy", "Accuracy", "higher"),
    ("roc_auc", "ROC AUC", "higher"),
)


def load_model_summaries(reports_dir: Path) -> pd.DataFrame:
    """Collect outer-fold metrics from every individual Step 08 JSON report."""
    rows: list[dict[str, Any]] = []
    for report_path in sorted(reports_dir.glob("*_nested_walk_forward.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = report.get("overall_out_of_sample_metrics", {})
        if not report.get("model") or not all(metric in metrics for metric, _, _ in METRICS):
            continue
        rows.append(
            {
                "model": report["model"],
                "feature_count": report.get("feature_count"),
                "training_window_years": report.get("training_window_years"),
                "evaluation_years": ", ".join(
                    str(year) for year in report.get("evaluation_years", [])
                ),
                "report_modified_utc": datetime.fromtimestamp(
                    report_path.stat().st_mtime, timezone.utc
                ).isoformat(timespec="seconds"),
                **{metric: float(metrics[metric]) for metric, _, _ in METRICS},
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"No Step 08 tuning reports were found in {reports_dir}. "
            "Run step_08_hyperparameter_tuning.py first."
        )
    return pd.DataFrame(rows).sort_values("log_loss", kind="stable").reset_index(
        drop=True
    )


def ranked_summary(summaries: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates without collapsing the individual probability metrics."""
    ranked = summaries.copy()
    for metric, _, direction in METRICS:
        ranked[f"{metric}_rank"] = ranked[metric].rank(
            method="min", ascending=direction == "lower"
        ).astype(int)
    rank_columns = [f"{metric}_rank" for metric, _, _ in METRICS]
    ranked["mean_metric_rank"] = ranked[rank_columns].mean(axis=1).round(2)
    ranked["log_loss_leader"] = ranked["log_loss_rank"].eq(1)
    return ranked.sort_values(
        ["log_loss_rank", "brier_score_rank", "mean_metric_rank"], kind="stable"
    ).reset_index(drop=True)


def model_metrics_figure(summary: pd.DataFrame) -> go.Figure:
    """Show held-out quality metrics without placing incompatible scales together."""
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Log loss — lower is better",
            "Brier score — lower is better",
            "Accuracy — higher is better",
            "ROC AUC — higher is better",
        ],
    )
    for position, (metric, label, _) in enumerate(METRICS):
        row = position // 2 + 1
        column = position % 2 + 1
        figure.add_trace(
            go.Bar(
                x=summary["model"],
                y=summary[metric],
                text=summary[metric].map(lambda value: f"{value:.3f}"),
                textposition="outside",
                marker_color="#2563eb" if metric in {"log_loss", "brier_score"} else "#059669",
                name=label,
                showlegend=False,
                hovertemplate=f"Model: %{{x}}<br>{label}: %{{y:.4f}}<extra></extra>",
            ),
            row=row,
            col=column,
        )
    figure.update_layout(
        title="Step 08 held-out model comparison",
        template="plotly_white",
        height=700,
        margin={"t": 90},
    )
    return figure


def load_candidate_results(
    reports_dir: Path, model_names: pd.Series
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load candidate-level metrics where Step 08 generated multi-metric CSVs."""
    required = {
        "evaluation_year",
        "candidate_rank",
        "mean_inner_log_loss",
        "mean_inner_accuracy",
        "mean_inner_brier_score",
        "mean_fit_seconds",
        "parameters",
    }
    available: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    for model_name in model_names:
        path = reports_dir / f"{model_name}_inner_grid_results.csv"
        if not path.is_file():
            warnings.append(f"{model_name}: candidate-results CSV is missing.")
            continue
        candidates = pd.read_csv(path)
        missing = sorted(required - set(candidates.columns))
        if missing:
            warnings.append(
                f"{model_name}: rerun Step 08 to add candidate accuracy/Brier metrics. "
                f"Missing: {', '.join(missing)}."
            )
            continue
        available[model_name] = candidates
    return available, warnings


def candidate_tradeoff_figure(candidates_by_model: dict[str, pd.DataFrame]) -> go.Figure:
    """Compare every sampled/grid candidate by inner log loss and accuracy."""
    models = list(candidates_by_model)
    figure = make_subplots(
        rows=1,
        cols=len(models),
        subplot_titles=[model.replace("_", " ").title() for model in models],
        shared_yaxes=True,
    )
    colors = ["#2563eb", "#7c3aed", "#ea580c", "#059669", "#db2777", "#0891b2"]
    for column, model_name in enumerate(models, start=1):
        candidates = candidates_by_model[model_name]
        for color_index, (year, rows) in enumerate(
            candidates.groupby("evaluation_year", sort=True)
        ):
            figure.add_trace(
                go.Scatter(
                    x=rows["mean_inner_log_loss"],
                    y=rows["mean_inner_accuracy"],
                    mode="markers",
                    name=f"{model_name} · {year}",
                    marker={"size": 9, "opacity": 0.82, "color": colors[color_index % len(colors)]},
                    customdata=rows[
                        [
                            "candidate_rank",
                            "mean_inner_brier_score",
                            "mean_fit_seconds",
                            "parameters",
                        ]
                    ].to_numpy(),
                    hovertemplate=(
                        "Inner log loss: %{x:.4f}<br>Inner accuracy: %{y:.2%}"
                        "<br>Candidate rank: %{customdata[0]}"
                        "<br>Inner Brier score: %{customdata[1]:.4f}"
                        "<br>Mean fit time: %{customdata[2]:.1f}s"
                        "<br>Parameters: %{customdata[3]}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(
            title_text="Inner log loss<br>(lower is better)", row=1, col=column
        )
    figure.update_yaxes(title_text="Inner accuracy (higher is better)", tickformat=".0%")
    figure.update_layout(
        title="Step 08 parameter candidates",
        template="plotly_white",
        height=520,
        width=max(700, 420 * len(models)),
    )
    return figure


def calibration_figure(reports_dir: Path, model_names: pd.Series) -> tuple[go.Figure, list[str]]:
    """Overlay each finalist's outer-fold reliability curve."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"color": "#94a3b8", "dash": "dash"},
        )
    )
    warnings: list[str] = []
    for model_name in model_names:
        path = reports_dir / f"{model_name}_calibration.csv"
        if not path.is_file():
            warnings.append(f"{model_name}: calibration CSV is missing.")
            continue
        calibration = pd.read_csv(path)
        visible = calibration.loc[calibration["match_count"].gt(0)]
        if visible.empty:
            warnings.append(f"{model_name}: calibration CSV has no populated bins.")
            continue
        figure.add_trace(
            go.Scatter(
                x=visible["mean_predicted_probability"],
                y=visible["observed_win_rate"],
                mode="lines+markers",
                name=model_name,
                marker={"size": 8},
                customdata=visible[
                    ["bin_lower", "bin_upper", "match_count", "absolute_calibration_gap"]
                ].to_numpy(),
                hovertemplate=(
                    "Predicted: %{x:.1%}<br>Observed: %{y:.1%}"
                    "<br>Bin: %{customdata[0]:.0%}–%{customdata[1]:.0%}"
                    "<br>Matches: %{customdata[2]}"
                    "<br>Calibration gap: %{customdata[3]:.1%}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Outer-fold reliability comparison",
        xaxis={"title": "Mean predicted win probability", "range": [0, 1], "tickformat": ".0%"},
        yaxis={"title": "Observed win rate", "range": [0, 1], "tickformat": ".0%"},
        template="plotly_white",
    )
    return figure, warnings


def write_html_hub(
    output_path: Path,
    summary: pd.DataFrame,
    figures: list[go.Figure],
    warnings: list[str],
) -> None:
    """Write one convenient, self-contained model-selection report."""
    warning_html = "".join(f"<li>{warning}</li>" for warning in warnings)
    warning_section = (
        f"<section><h2>Refresh needed</h2><ul>{warning_html}</ul></section>"
        if warnings
        else ""
    )
    figure_html = [
        figure.to_html(full_html=False, include_plotlyjs="cdn" if index == 0 else False)
        for index, figure in enumerate(figures)
    ]
    leader = summary.iloc[0]
    table_html = summary.to_html(index=False, float_format=lambda value: f"{value:.4f}")
    output_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Step 09 final model choice</title>"
        "<style>body{font-family:Arial,sans-serif;margin:32px;color:#1f2937;}"
        "section{margin:32px 0;}table{border-collapse:collapse;width:100%;font-size:13px;}"
        "th,td{border:1px solid #d1d5db;padding:8px;text-align:left;}"
        "th{background:#eff6ff;}li{margin:6px 0;}</style></head><body>"
        "<h1>Step 09 — Final model choice</h1>"
        f"<p>Current log-loss leader: <strong>{leader['model']}</strong> "
        f"({leader['log_loss']:.4f}). Confirm calibration and Brier score before "
        "choosing the final production model.</p>"
        "<section><h2>Decision table</h2>"
        "<p>Ranks are supporting evidence; choose primarily on held-out log loss, "
        "then calibration and Brier score—not accuracy alone.</p>"
        f"{table_html}</section>{warning_section}"
        + "".join(f"<section>{html}</section>" for html in figure_html)
        + "</body></html>",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Define the read-only model-choice report command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = ranked_summary(load_model_summaries(args.reports_dir))
    candidate_results, candidate_warnings = load_candidate_results(
        args.reports_dir, summary["model"]
    )
    calibration, calibration_warnings = calibration_figure(
        args.reports_dir, summary["model"]
    )
    warnings = [*candidate_warnings, *calibration_warnings]
    if summary["evaluation_years"].nunique(dropna=False) > 1:
        warnings.append(
            "Candidates were evaluated over different year ranges. Rerun Step 08 for all "
            "models with the same evaluation years before making a final selection."
        )
    figures = [model_metrics_figure(summary), calibration]
    if candidate_results:
        figures.insert(1, candidate_tradeoff_figure(candidate_results))
    else:
        warnings.append("No current candidate-level accuracy results were available.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "final_model_choice_summary.csv"
    report_path = args.output_dir / "final_model_choice.html"
    summary.to_csv(summary_path, index=False)
    write_html_hub(report_path, summary, figures, warnings)

    leader = summary.iloc[0]
    print(f"Current log-loss leader: {leader['model']} ({leader['log_loss']:.4f})")
    print(f"Wrote decision summary to {summary_path}")
    print(f"Wrote interactive model-choice report to {report_path}")
    for warning in warnings:
        print(f"Note: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
