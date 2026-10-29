"""Interactive source-quality explorer for the WTA match datasets.

Run from this project directory:
    streamlit run step_04_data_explorer_dashboard.py
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.express as px
import streamlit as st

from environmental_data_pipeline import (
    ENVIRONMENT_COLUMNS,
    TOURNAMENT_WEEK_WEATHER_PATH,
    enrich_matches_with_environment,
)
from step_02_load_and_compatre_providers import load_provider


st.set_page_config(page_title="WTA data explorer", page_icon="🎾", layout="wide")

SOURCE_SPECS = {
    "Sackmann": ("sackmann", "wta_matches_*.csv"),
    "TennisMyLife": ("tennismylife", "*_wta.csv"),
}
KEY_COLUMNS = [
    "tourney_date",
    "tourney_name",
    "winner_name",
    "loser_name",
    "score",
    "surface",
    "winner_rank",
    "loser_rank",
    "w_svpt",
    "l_svpt",
]
COMPARISON_COLUMNS = ["winner_name", "loser_name", "score", "surface", "round"]
IDENTITY_COLUMNS = [
    "tourney_date",
    "tourney_name",
    "round",
    "winner_name",
    "loser_name",
]
SERVE_STAT_COLUMNS = [
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
]
QUALITY_COLUMNS = list(dict.fromkeys(KEY_COLUMNS + SERVE_STAT_COLUMNS))
PROVIDER_TIE_BREAK = {"TennisMyLife": 1, "Sackmann": 0}
INCOMPLETE_SCORE_PATTERN = r"\b(?:RET|W/O|WALKOVER|DEF|ABN|ABD)\b"


def normalized_identity_part(series: pd.Series) -> pd.Series:
    """Normalize text components used to identify the same match across sources."""
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def environmental_cache_token() -> int:
    """Invalidate Streamlit's data cache whenever cached weather is rebuilt."""
    try:
        return TOURNAMENT_WEEK_WEATHER_PATH.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


@st.cache_data(show_spinner="Loading source files…")
def load_all_matches(environment_version: int) -> pd.DataFrame:
    """Load both raw sources and add dashboard-only quality fields."""
    del environment_version
    frames = []
    for display_name, (provider, pattern) in SOURCE_SPECS.items():
        frame = load_provider(provider, pattern).copy()
        frame["provider_label"] = display_name
        frames.append(frame)

    matches = enrich_matches_with_environment(pd.concat(frames, ignore_index=True))
    matches["match_date"] = pd.to_datetime(
        matches["tourney_date"].astype("string"), format="%Y%m%d", errors="coerce"
    )
    matches["score_normalized"] = matches["score"].fillna("").astype(str).str.strip().str.upper()
    matches["match_status"] = "Completed"
    matches.loc[matches["score_normalized"].eq(""), "match_status"] = "Unknown"
    matches.loc[
        matches["score_normalized"].str.contains(INCOMPLETE_SCORE_PATTERN, regex=True),
        "match_status",
    ] = "Incomplete"

    winner_rank = pd.to_numeric(matches["winner_rank"], errors="coerce")
    loser_rank = pd.to_numeric(matches["loser_rank"], errors="coerce")
    matches["both_rankings_available"] = (winner_rank > 0) & (loser_rank > 0)

    winner_serve_points = pd.to_numeric(matches["w_svpt"], errors="coerce")
    loser_serve_points = pd.to_numeric(matches["l_svpt"], errors="coerce")
    matches["both_serve_stats_available"] = (
        (winner_serve_points > 0) & (loser_serve_points > 0)
    )
    matches["match_key"] = (
        matches["tourney_id"].astype("string")
        + " | "
        + matches["match_num"].astype("string")
    )
    identity_parts = [
        matches["tourney_date"].astype("string"),
        *[normalized_identity_part(matches[column]) for column in IDENTITY_COLUMNS[1:]],
    ]
    matches["match_identity"] = identity_parts[0]
    for part in identity_parts[1:]:
        matches["match_identity"] += " | " + part
    matches["record_completeness"] = matches[QUALITY_COLUMNS].notna().sum(axis=1)
    matches["provider_tie_break"] = matches["provider_label"].map(PROVIDER_TIE_BREAK)
    return matches


def pct(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if not denominator else 100 * numerator / denominator


def availability_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one quality-summary row for each provider in a frame."""
    return (
        frame.groupby("provider_label", dropna=False)
        .agg(
            rows=("provider", "size"),
            incomplete=("match_status", lambda status: (status == "Incomplete").sum()),
            unknown_score=("match_status", lambda status: (status == "Unknown").sum()),
            both_rankings=("both_rankings_available", "sum"),
            both_serve_stats=("both_serve_stats_available", "sum"),
        )
        .reset_index()
        .assign(
            incomplete_pct=lambda x: (100 * x["incomplete"] / x["rows"]).round(2),
            rankings_available_pct=lambda x: (100 * x["both_rankings"] / x["rows"]).round(2),
            serve_stats_available_pct=lambda x: (
                100 * x["both_serve_stats"] / x["rows"]
            ).round(2),
        )
    )


def missingness_by_provider(
    frame: pd.DataFrame, columns: Iterable[str]
) -> pd.DataFrame:
    records = []
    for provider, provider_frame in frame.groupby("provider_label"):
        for column in columns:
            records.append(
                {
                    "Provider": provider,
                    "Field": column,
                    "Missing (%)": round(provider_frame[column].isna().mean() * 100, 2),
                    "Missing rows": int(provider_frame[column].isna().sum()),
                }
            )
    return pd.DataFrame(records)


def select_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Render filters and return the corresponding rows."""
    st.sidebar.header("Filter matches")
    provider_options = list(SOURCE_SPECS)
    providers = st.sidebar.multiselect(
        "Providers", provider_options, default=provider_options
    )
    years = sorted(frame["source_year"].dropna().astype(int).unique().tolist())
    selected_years = st.sidebar.slider(
        "Source year", min_value=min(years), max_value=max(years), value=(min(years), max(years))
    )
    status_options = ["Completed", "Incomplete", "Unknown"]
    statuses = st.sidebar.multiselect(
        "Match status", status_options, default=status_options
    )
    surfaces = sorted(frame["surface"].fillna("(missing)").astype(str).unique().tolist())
    selected_surfaces = st.sidebar.multiselect("Surface", surfaces, default=surfaces)
    levels = sorted(
        frame["tourney_level"].fillna("(missing)").astype(str).unique().tolist()
    )
    selected_levels = st.sidebar.multiselect("Tournament level", levels, default=levels)

    result = frame[
        frame["provider_label"].isin(providers)
        & frame["source_year"].between(*selected_years)
        & frame["match_status"].isin(statuses)
        & frame["surface"].fillna("(missing)").astype(str).isin(selected_surfaces)
        & frame["tourney_level"].fillna("(missing)").astype(str).isin(selected_levels)
    ].copy()
    st.sidebar.caption(f"{len(result):,} matches selected")
    return result


def draw_overview(frame: pd.DataFrame) -> None:
    st.subheader("Quality overview")
    if frame.empty:
        st.info("No matches match the current filters.")
        return

    rows = len(frame)
    incomplete = (frame["match_status"] == "Incomplete").sum()
    weather_available = frame["weather_status"].eq("ok")
    cols = st.columns(5)
    cols[0].metric("Matches", f"{rows:,}")
    cols[1].metric("Incomplete matches", f"{pct(incomplete, rows):.2f}%")
    cols[2].metric(
        "Both rankings available", f"{frame['both_rankings_available'].mean() * 100:.2f}%"
    )
    cols[3].metric(
        "Both serve-stat sets available",
        f"{frame['both_serve_stats_available'].mean() * 100:.2f}%",
    )
    cols[4].metric(
        "Tournament-week weather available", f"{weather_available.mean() * 100:.2f}%"
    )

    summary = availability_summary(frame)
    display_summary = summary.rename(
        columns={
            "provider_label": "Provider",
            "rows": "Matches",
            "incomplete": "Incomplete",
            "unknown_score": "Unknown score",
            "incomplete_pct": "Incomplete (%)",
            "rankings_available_pct": "Rankings available (%)",
            "serve_stats_available_pct": "Serve stats available (%)",
        }
    )[
        [
            "Provider",
            "Matches",
            "Incomplete",
            "Incomplete (%)",
            "Rankings available (%)",
            "Serve stats available (%)",
            "Unknown score",
        ]
    ]
    st.dataframe(display_summary, hide_index=True, width="stretch")

    yearly = (
        frame.groupby(["source_year", "provider_label"], as_index=False)
        .agg(
            matches=("provider", "size"),
            incomplete_rate=(
                "match_status",
                lambda status: pct((status == "Incomplete").sum(), len(status)),
            ),
            ranking_coverage=("both_rankings_available", "mean"),
            serve_coverage=("both_serve_stats_available", "mean"),
        )
        .assign(
            ranking_coverage=lambda x: x["ranking_coverage"] * 100,
            serve_coverage=lambda x: x["serve_coverage"] * 100,
        )
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(
                yearly,
                x="source_year",
                y="matches",
                color="provider_label",
                barmode="group",
                labels={
                    "source_year": "Source year",
                    "matches": "Matches",
                    "provider_label": "Provider",
                },
                title="Records by source year",
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.line(
                yearly,
                x="source_year",
                y="incomplete_rate",
                color="provider_label",
                markers=True,
                labels={
                    "source_year": "Source year",
                    "incomplete_rate": "Incomplete matches (%)",
                    "provider_label": "Provider",
                },
                title="Incomplete-match rate",
            ).update_yaxes(ticksuffix="%", rangemode="tozero"),
            width="stretch",
        )

    coverage = yearly.melt(
        id_vars=["source_year", "provider_label"],
        value_vars=["ranking_coverage", "serve_coverage"],
        var_name="Measure",
        value_name="Coverage (%)",
    ).replace(
        {"Measure": {"ranking_coverage": "Both rankings", "serve_coverage": "Both serve-stat sets"}}
    )
    st.plotly_chart(
        px.line(
            coverage,
            x="source_year",
            y="Coverage (%)",
            color="provider_label",
            line_dash="Measure",
            markers=True,
            labels={"source_year": "Source year", "provider_label": "Provider"},
            title="Ranking and serve-stat coverage",
        ).update_yaxes(ticksuffix="%", range=[0, 101]),
        width="stretch",
    )


def draw_missingness(frame: pd.DataFrame) -> None:
    st.subheader("Missingness and match completion")
    if frame.empty:
        st.info("No matches match the current filters.")
        return

    field_groups = {
        "Core match fields": KEY_COLUMNS,
        "Serve statistics": [
            "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
            "w_SvGms", "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt",
            "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved",
            "l_bpFaced",
        ],
        "Player and rankings": [
            "winner_id", "loser_id", "winner_name", "loser_name", "winner_age",
            "loser_age", "winner_rank", "loser_rank", "winner_rank_points",
            "loser_rank_points",
        ],
        "Environmental enrichment": ENVIRONMENT_COLUMNS,
    }
    field_group_name = st.radio(
        "Fields to inspect",
        list(field_groups),
        horizontal=True,
    )
    field_group = field_groups[field_group_name]
    available_fields = [column for column in field_group if column in frame]
    missingness = missingness_by_provider(frame, available_fields)
    st.plotly_chart(
        px.bar(
            missingness,
            x="Missing (%)",
            y="Field",
            color="Provider",
            orientation="h",
            barmode="group",
            labels={"Missing (%)": "Missing values (%)"},
            title="Missing values by field and provider",
        ).update_xaxes(ticksuffix="%", rangemode="tozero"),
        width="stretch",
    )

    status_summary = (
        frame.groupby(["provider_label", "match_status"], as_index=False)
        .size()
        .rename(columns={"size": "matches"})
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(
                status_summary,
                x="provider_label",
                y="matches",
                color="match_status",
                barmode="stack",
                category_orders={"match_status": ["Completed", "Incomplete", "Unknown"]},
                labels={
                    "provider_label": "Provider",
                    "matches": "Matches",
                    "match_status": "Status",
                },
                title="Match status counts",
            ),
            width="stretch",
        )
    with right:
        completion_missing = (
            frame.groupby(["provider_label", "match_status"], as_index=False)
            .agg(
                ranking_missing=(
                    "both_rankings_available", lambda available: 100 * (1 - available.mean())
                ),
                serve_missing=(
                    "both_serve_stats_available", lambda available: 100 * (1 - available.mean())
                ),
            )
            .melt(
                id_vars=["provider_label", "match_status"],
                var_name="Measure",
                value_name="Missing (%)",
            )
            .replace(
                {
                    "Measure": {
                        "ranking_missing": "At least one ranking missing",
                        "serve_missing": "At least one serve-stat set missing",
                    }
                }
            )
        )
        st.plotly_chart(
            px.bar(
                completion_missing,
                x="match_status",
                y="Missing (%)",
                color="Measure",
                facet_col="provider_label",
                barmode="group",
                category_orders={"match_status": ["Completed", "Incomplete", "Unknown"]},
                labels={"match_status": "Status", "provider_label": "Provider"},
                title="Field availability by completion status",
            ).update_yaxes(ticksuffix="%", rangemode="tozero"),
            width="stretch",
        )

    st.dataframe(
        missingness.sort_values(["Missing (%)", "Field"], ascending=[False, True]),
        hide_index=True,
        width="stretch",
    )


def comparable_source_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Use a consistent subset and make one record per provider/match identity."""
    comparison = frame[frame["provider_label"].isin(SOURCE_SPECS)].copy()
    return comparison.drop_duplicates(
        subset=["provider_label", "match_identity"], keep="first"
    )


def draw_source_comparison(frame: pd.DataFrame) -> None:
    st.subheader("Cross-source comparison")
    comparison = comparable_source_rows(frame)
    sources_present = set(comparison["provider_label"])
    if sources_present != set(SOURCE_SPECS):
        st.info("Select both providers to compare matched records.")
        return

    sackmann = comparison[comparison["provider_label"] == "Sackmann"]
    tml = comparison[comparison["provider_label"] == "TennisMyLife"]
    merged = sackmann.merge(
        tml,
        on="match_identity",
        how="outer",
        suffixes=("_sackmann", "_tml"),
        indicator=True,
    )
    matched = merged[merged["_merge"] == "both"].copy()

    def fields_that_differ(row: pd.Series) -> str:
        differences = []
        for column in COMPARISON_COLUMNS:
            left = row.get(f"{column}_sackmann")
            right = row.get(f"{column}_tml")
            if pd.isna(left) and pd.isna(right):
                continue
            if str(left).strip().upper() != str(right).strip().upper():
                differences.append(column)
        return ", ".join(differences) if differences else "None"

    matched["differing_fields"] = matched.apply(fields_that_differ, axis=1)
    same_fields = (matched["differing_fields"] == "None").sum()
    metrics = st.columns(4)
    metrics[0].metric("Unique Sackmann matches", f"{len(sackmann):,}")
    metrics[1].metric("Unique TennisMyLife matches", f"{len(tml):,}")
    metrics[2].metric("Matches in both", f"{len(matched):,}")
    metrics[3].metric("Same core fields", f"{pct(same_fields, len(matched)):.2f}%")

    presence = merged["_merge"].map(
        {
            "left_only": "Sackmann only",
            "right_only": "TennisMyLife only",
            "both": "Both sources",
        }
    ).value_counts().rename_axis("Presence").reset_index(name="Records")
    left, right = st.columns([1, 2])
    with left:
        st.plotly_chart(
            px.pie(
                presence,
                names="Presence",
                values="Records",
                title="Match-identity coverage",
                hole=0.45,
            ),
            width="stretch",
        )
    with right:
        difference_counts = (
            matched.assign(has_difference=matched["differing_fields"] != "None")
            .groupby("has_difference", as_index=False)
            .size()
            .replace({"has_difference": {False: "All checked fields agree", True: "At least one field differs"}})
            .rename(columns={"size": "Records", "has_difference": "Comparison"})
        )
        st.plotly_chart(
            px.bar(
                difference_counts,
                x="Comparison",
                y="Records",
                text="Records",
                title="Core-field agreement among shared keys",
            ),
            width="stretch",
        )

    differences = matched[matched["differing_fields"] != "None"].copy()
    st.caption(
        "Matches are identified by tournament week, normalized tournament name, round, winner, and loser. "
        "Agreement checks winner, loser, score, surface, and round."
    )
    st.dataframe(
        differences[
            [
                "match_identity",
                "tourney_name_sackmann",
                "winner_name_sackmann",
                "loser_name_sackmann",
                "score_sackmann",
                "score_tml",
                "differing_fields",
            ]
        ].rename(
            columns={
                "match_identity": "Match identity",
                "tourney_name_sackmann": "Tournament",
                "winner_name_sackmann": "Winner",
                "loser_name_sackmann": "Loser",
                "score_sackmann": "Sackmann score",
                "score_tml": "TennisMyLife score",
                "differing_fields": "Differing fields",
            }
        ),
        hide_index=True,
        width="stretch",
        height=360,
    )


def combine_source_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain one best-quality raw record for every cross-source match identity."""
    ordered = frame.sort_values(
        ["match_identity", "record_completeness", "provider_tie_break"],
        ascending=[True, False, False],
        kind="stable",
    )
    source_metadata = (
        ordered.groupby("match_identity", as_index=False)
        .agg(
            source_record_count=("provider_label", "size"),
            source_providers=(
                "provider_label",
                lambda providers: ", ".join(sorted(set(providers))),
            ),
        )
    )
    combined = ordered.drop_duplicates("match_identity", keep="first").copy()
    combined = combined.merge(source_metadata, on="match_identity", how="left")
    combined["retained_provider"] = combined["provider_label"]
    combined["provider"] = "combined"
    combined["provider_label"] = "Combined"
    combined["is_cross_source_duplicate"] = combined["source_record_count"].gt(1)
    if combined["match_identity"].duplicated().any():
        raise RuntimeError("The combined dataset contains duplicate match identities.")
    return combined


def draw_combined_source(frame: pd.DataFrame) -> None:
    st.subheader("Combined, de-duplicated source")
    if frame.empty:
        st.info("No matches match the current filters.")
        return
    if set(frame["provider_label"]) != set(SOURCE_SPECS):
        st.info("Select both providers to create a combined, de-duplicated source.")
        return

    combined = combine_source_rows(frame)
    raw_rows = len(frame)
    duplicate_rows_removed = raw_rows - len(combined)
    duplicate_groups = int(combined["is_cross_source_duplicate"].sum())

    metrics = st.columns(4)
    metrics[0].metric("Input source rows", f"{raw_rows:,}")
    metrics[1].metric("Unique combined matches", f"{len(combined):,}")
    metrics[2].metric("Cross-source duplicates removed", f"{duplicate_rows_removed:,}")
    metrics[3].metric("Duplicate identities remaining", "0")

    st.caption(
        "A match is identified by tournament week, normalized tournament name, round, winner, "
        "and loser. When both sources contain that match, the record with the most populated "
        "quality fields is retained; ties favor TennisMyLife because it includes the extra `indoor` field."
    )

    left, right = st.columns(2)
    with left:
        source_breakdown = (
            combined.assign(
                source_coverage=combined["source_record_count"].map(
                    {1: "One source", 2: "Both sources"}
                )
            )
            .groupby(["source_year", "source_coverage"], as_index=False)
            .size()
            .rename(columns={"size": "matches"})
        )
        st.plotly_chart(
            px.bar(
                source_breakdown,
                x="source_year",
                y="matches",
                color="source_coverage",
                barmode="stack",
                labels={
                    "source_year": "Source year",
                    "matches": "Unique matches",
                    "source_coverage": "Available from",
                },
                title="Combined matches by source coverage",
            ),
            width="stretch",
        )
    with right:
        retained_breakdown = (
            combined.groupby("retained_provider", as_index=False)
            .size()
            .rename(columns={"size": "matches", "retained_provider": "Provider retained"})
        )
        st.plotly_chart(
            px.bar(
                retained_breakdown,
                x="Provider retained",
                y="matches",
                text="matches",
                labels={"matches": "Unique matches"},
                title="Record retained after consolidation",
            ),
            width="stretch",
        )

    retained_columns = [
        "source_year",
        "tourney_name",
        "round",
        "winner_name",
        "loser_name",
        "score",
        "retained_provider",
        "source_providers",
        "source_record_count",
        "record_completeness",
    ]
    st.caption(
        f"{duplicate_groups:,} combined matches were supplied by both sources. "
        "Every row below has one unique match identity."
    )
    st.dataframe(
        combined[retained_columns]
        .sort_values(["source_year", "tourney_name", "round"], ascending=[False, True, True])
        .rename(
            columns={
                "source_year": "Year",
                "tourney_name": "Tournament",
                "winner_name": "Winner",
                "loser_name": "Loser",
                "retained_provider": "Retained record",
                "source_providers": "Available from",
                "source_record_count": "Source records merged",
                "record_completeness": "Quality fields populated",
            }
        ),
        hide_index=True,
        width="stretch",
        height=400,
    )

    export_columns = [
        column
        for column in combined.columns
        if column not in {"score_normalized", "provider_tie_break"}
    ]
    st.download_button(
        "Download combined CSV",
        data=combined[export_columns].to_csv(index=False).encode("utf-8"),
        file_name="wta_matches_combined_deduplicated.csv",
        mime="text/csv",
    )


def draw_environmental_data(frame: pd.DataFrame) -> None:
    st.subheader("Tournament-week environmental data")
    if frame.empty:
        st.info("No matches match the current filters.")
        return

    venue_mapped = frame["venue_match_status"].eq("matched")
    weather_available = frame["weather_status"].eq("ok")
    metrics = st.columns(4)
    metrics[0].metric("Matches", f"{len(frame):,}")
    metrics[1].metric("Venue mapped", f"{venue_mapped.mean() * 100:.2f}%")
    metrics[2].metric("Weather available", f"{weather_available.mean() * 100:.2f}%")
    metrics[3].metric(
        "Unique tournament weeks with weather",
        f"{frame.loc[weather_available, 'tournament_week_key'].nunique():,}",
    )
    st.info(
        "These are seven-day, venue- or city-level weather aggregates from the tournament start date. "
        "They are not match-day observations and should not be used as pre-match model features."
    )

    coverage = (
        frame.assign(
            Venue_mapped=venue_mapped,
            Weather_available=weather_available,
        )
        .groupby(["source_year", "provider_label"], as_index=False)
        .agg(
            Venue_mapped=("Venue_mapped", "mean"),
            Weather_available=("Weather_available", "mean"),
        )
        .melt(
            id_vars=["source_year", "provider_label"],
            var_name="Measure",
            value_name="Coverage (%)",
        )
        .assign(**{"Coverage (%)": lambda x: x["Coverage (%)"] * 100})
    )
    st.plotly_chart(
        px.line(
            coverage,
            x="source_year",
            y="Coverage (%)",
            color="provider_label",
            line_dash="Measure",
            markers=True,
            labels={"source_year": "Source year", "provider_label": "Provider"},
            title="Venue-mapping and weather coverage",
        ).update_yaxes(ticksuffix="%", range=[0, 101]),
        width="stretch",
    )

    weekly_weather = frame.loc[weather_available].drop_duplicates("tournament_week_key")
    weather_columns = [
        "tourney_name",
        "tourney_date",
        "venue_name",
        "venue_city",
        "venue_country",
        "venue_indoor_override",
        "venue_mapping_method",
        "venue_location_precision",
        "weather_temperature_mean_f",
        "weather_temperature_max_f",
        "weather_precipitation_total_mm",
        "weather_wind_speed_max_kmh",
        "weather_wind_gusts_max_kmh",
        "weather_sunshine_duration_hours",
        "weather_status",
    ]
    available_weather_columns = [
        column for column in weather_columns if column in weekly_weather
    ]
    st.dataframe(
        weekly_weather[available_weather_columns]
        .sort_values(["tourney_date", "tourney_name"], ascending=[False, True])
        .rename(
            columns={
                "tourney_name": "Tournament",
                "tourney_date": "Tournament week start",
                "venue_name": "Location",
                "venue_city": "City",
                "venue_country": "Country",
                "venue_indoor_override": "Indoor",
                "venue_mapping_method": "Location source",
                "venue_location_precision": "Location precision",
                "weather_temperature_mean_f": "Mean temperature (°F)",
                "weather_temperature_max_f": "Maximum temperature (°F)",
                "weather_precipitation_total_mm": "Precipitation (mm)",
                "weather_wind_speed_max_kmh": "Maximum wind (km/h)",
                "weather_wind_gusts_max_kmh": "Maximum gust (km/h)",
                "weather_sunshine_duration_hours": "Sunshine (hours)",
                "weather_status": "Weather status",
            }
        ),
        hide_index=True,
        width="stretch",
        height=360,
    )


def player_match_log(frame: pd.DataFrame, player: str) -> pd.DataFrame:
    """Make player-relative results and service fields from de-duplicated matches."""
    matches = frame[
        (frame["winner_name"] == player) | (frame["loser_name"] == player)
    ].copy()
    won = matches["winner_name"].eq(player)
    log = matches[
        [
            "source_year",
            "tourney_date",
            "match_date",
            "tourney_name",
            "round",
            "surface",
            "score",
            "match_status",
            "minutes",
            "weather_temperature_mean_f",
            "weather_wind_speed_max_kmh",
            "weather_status",
        ]
    ].copy()
    log["result"] = won.map({True: "Win", False: "Loss"})
    log["opponent"] = matches["loser_name"].where(won, matches["winner_name"])
    log["player_rank"] = pd.to_numeric(
        matches["winner_rank"].where(won, matches["loser_rank"]), errors="coerce"
    )
    log["opponent_rank"] = pd.to_numeric(
        matches["loser_rank"].where(won, matches["winner_rank"]), errors="coerce"
    )

    service_fields = {
        "aces": "ace",
        "double_faults": "df",
        "serve_points": "svpt",
        "first_serves_in": "1stIn",
        "first_serve_points_won": "1stWon",
        "second_serve_points_won": "2ndWon",
        "service_games": "SvGms",
        "break_points_saved": "bpSaved",
        "break_points_faced": "bpFaced",
    }
    for output_column, source_suffix in service_fields.items():
        values = matches[f"w_{source_suffix}"].where(won, matches[f"l_{source_suffix}"])
        log[output_column] = pd.to_numeric(values, errors="coerce")

    def rate(numerator: str, denominator: str) -> pd.Series:
        denominator_values = log[denominator].replace(0, pd.NA)
        return 100 * log[numerator] / denominator_values

    log["first_serve_in_pct"] = rate("first_serves_in", "serve_points")
    log["first_serve_won_pct"] = rate("first_serve_points_won", "first_serves_in")
    log["second_serve_points"] = log["serve_points"] - log["first_serves_in"]
    log["second_serve_won_pct"] = rate("second_serve_points_won", "second_serve_points")
    log["ace_rate_pct"] = rate("aces", "serve_points")
    log["double_fault_rate_pct"] = rate("double_faults", "serve_points")
    log["break_points_saved_pct"] = rate("break_points_saved", "break_points_faced")
    return log


def draw_player_study(frame: pd.DataFrame, player: str) -> None:
    log = player_match_log(frame, player)
    if log.empty:
        st.info("No matches for this player match the current filters.")
        return

    st.subheader(f"Player study: {player}")
    wins = int(log["result"].eq("Win").sum())
    completed = log["match_status"].eq("Completed")
    metrics = st.columns(5)
    metrics[0].metric("Combined matches", f"{len(log):,}")
    metrics[1].metric("Record", f"{wins}-{len(log) - wins}")
    metrics[2].metric("Win rate", f"{wins / len(log) * 100:.1f}%")
    metrics[3].metric("Completed matches", f"{completed.mean() * 100:.1f}%")
    metrics[4].metric("Serve stats available", f"{log['serve_points'].notna().mean() * 100:.1f}%")

    yearly = (
        log.groupby(["source_year", "result"], as_index=False)
        .size()
        .rename(columns={"size": "matches"})
    )
    surface = (
        log.groupby("surface", dropna=False)
        .agg(matches=("result", "size"), wins=("result", lambda results: (results == "Win").sum()))
        .reset_index()
        .assign(win_rate_pct=lambda x: 100 * x["wins"] / x["matches"])
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(
                yearly,
                x="source_year",
                y="matches",
                color="result",
                barmode="stack",
                category_orders={"result": ["Win", "Loss"]},
                labels={"source_year": "Year", "matches": "Matches", "result": "Result"},
                title="Season record",
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.bar(
                surface,
                x="surface",
                y="win_rate_pct",
                text="matches",
                labels={"surface": "Surface", "win_rate_pct": "Win rate (%)", "matches": "Matches"},
                title="Win rate by surface (label = matches)",
            ).update_yaxes(ticksuffix="%", range=[0, 100]),
            width="stretch",
        )

    ranking_log = log.dropna(subset=["match_date", "player_rank"])
    service_by_year = (
        log.groupby("source_year", as_index=False)
        .agg(
            serve_points=("serve_points", "sum"),
            first_serves_in=("first_serves_in", "sum"),
            first_serve_points_won=("first_serve_points_won", "sum"),
            second_serve_points_won=("second_serve_points_won", "sum"),
        )
    )
    service_by_year["second_serve_points"] = (
        service_by_year["serve_points"] - service_by_year["first_serves_in"]
    )
    service_by_year["First serve won (%)"] = (
        100
        * service_by_year["first_serve_points_won"]
        / service_by_year["first_serves_in"].replace(0, pd.NA)
    )
    service_by_year["Second serve won (%)"] = (
        100
        * service_by_year["second_serve_points_won"]
        / service_by_year["second_serve_points"].replace(0, pd.NA)
    )
    service_trend = service_by_year.melt(
        id_vars="source_year",
        value_vars=["First serve won (%)", "Second serve won (%)"],
        var_name="Measure",
        value_name="Rate (%)",
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.scatter(
                ranking_log,
                x="match_date",
                y="player_rank",
                color="result",
                hover_data=["tourney_name", "opponent", "score"],
                labels={"match_date": "Tournament week", "player_rank": "Player rank", "result": "Result"},
                title="Rank entering each match",
            ).update_yaxes(autorange="reversed"),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.line(
                service_trend,
                x="source_year",
                y="Rate (%)",
                color="Measure",
                markers=True,
                labels={"source_year": "Year"},
                title="Seasonal serve-point win rates",
            ).update_yaxes(ticksuffix="%", range=[0, 100]),
            width="stretch",
        )

    detail_columns = [
        "source_year",
        "tourney_name",
        "round",
        "result",
        "opponent",
        "score",
        "surface",
        "player_rank",
        "opponent_rank",
        "minutes",
        "aces",
        "double_faults",
        "first_serve_in_pct",
        "first_serve_won_pct",
        "second_serve_won_pct",
        "weather_temperature_mean_f",
        "weather_wind_speed_max_kmh",
    ]
    st.dataframe(
        log[detail_columns]
        .sort_values(["source_year", "tourney_name"], ascending=[False, True])
        .rename(
            columns={
                "source_year": "Year",
                "tourney_name": "Tournament",
                "player_rank": "Player rank",
                "opponent_rank": "Opponent rank",
                "double_faults": "Double faults",
                "first_serve_in_pct": "1st serve in (%)",
                "first_serve_won_pct": "1st serve won (%)",
                "second_serve_won_pct": "2nd serve won (%)",
                "weather_temperature_mean_f": "Week temperature (°F)",
                "weather_wind_speed_max_kmh": "Week max wind (km/h)",
            }
        ),
        hide_index=True,
        width="stretch",
        height=430,
    )


def draw_match_finder(frame: pd.DataFrame) -> None:
    st.subheader("Match finder")
    status = st.multiselect(
        "Show statuses",
        ["Completed", "Incomplete", "Unknown"],
        default=["Completed", "Incomplete", "Unknown"],
    )
    available_matches = combine_source_rows(frame[frame["match_status"].isin(status)])
    player_names = sorted(
        set(available_matches["winner_name"].dropna())
        | set(available_matches["loser_name"].dropna())
    )
    player = st.selectbox(
        "Study a player",
        ["Select a player…", *player_names],
        help="Player metrics use one combined record per match, not duplicated provider rows.",
    )
    if player != "Select a player…":
        draw_player_study(available_matches, player)
        st.divider()

    query = st.text_input(
        "Search player, tournament, score, or source file",
        placeholder="e.g. Coco Gauff, Indian Wells, RET",
    ).strip()
    result = available_matches.copy()
    if query:
        searchable = ["winner_name", "loser_name", "tourney_name", "score", "source_file"]
        match_mask = pd.Series(False, index=result.index)
        for column in searchable:
            match_mask |= result[column].fillna("").astype(str).str.contains(
                query, case=False, regex=False
            )
        result = result[match_mask]

    st.caption(f"{len(result):,} matching combined records")
    display_columns = [
        "retained_provider", "source_providers", "source_year", "tourney_name", "tourney_date",
        "round", "winner_name", "loser_name", "score", "match_status", "surface", "winner_rank",
        "loser_rank", "w_svpt", "l_svpt", "source_file",
    ]
    st.dataframe(
        result[display_columns]
        .sort_values(["source_year", "tourney_date", "tourney_name"], ascending=[False, False, True])
        .rename(
            columns={
                "retained_provider": "Retained provider",
                "source_providers": "Available from",
                "source_year": "Year",
                "tourney_name": "Tournament",
                "tourney_date": "Tournament date",
                "winner_name": "Winner",
                "loser_name": "Loser",
                "match_status": "Status",
                "winner_rank": "Winner rank",
                "loser_rank": "Loser rank",
                "w_svpt": "Winner serve points",
                "l_svpt": "Loser serve points",
                "source_file": "Source file",
            }
        ),
        hide_index=True,
        width="stretch",
        height=520,
    )


def main() -> None:
    st.title("WTA data-quality explorer")
    st.caption(
        "Inspect source coverage, incomplete outcomes, field availability, and shared match records."
    )
    try:
        all_matches = load_all_matches(environmental_cache_token())
    except ValueError as error:
        st.error(f"Could not load the raw source files: {error}")
        st.stop()

    selected_matches = select_rows(all_matches)
    overview, missingness, comparison, combined, environmental, finder = st.tabs(
        [
            "Overview",
            "Missingness",
            "Source comparison",
            "Combined source",
            "Environmental data",
            "Match finder",
        ]
    )
    with overview:
        draw_overview(selected_matches)
    with missingness:
        draw_missingness(selected_matches)
    with comparison:
        draw_source_comparison(selected_matches)
    with combined:
        draw_combined_source(selected_matches)
    with environmental:
        draw_environmental_data(selected_matches)
    with finder:
        draw_match_finder(selected_matches)


if __name__ == "__main__":
    main()
