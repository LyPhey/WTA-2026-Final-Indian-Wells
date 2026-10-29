# from step_02_load_and_compatre_providers import load_provider

# This pytho file should answer these questions separately for Sackmann and TennisMyLife:

# 1. Are key fields missing?
# tourney_date, winner_name, loser_name, score, surface, rankings, serve statistics.

# sackmann = load_provider('sackmann', 'wta_matches_*.csv')
# tennismylife = load_provider('tennismylife', '*_wta.csv')

# print(sackmann)
# print(tennismylife)

# # to visualize the sackmann columns
# print("\nSackmann columns:")
# print(sackmann.columns.tolist())

# # to visualize the tennisislife columns
# print("\nTennisIsLife columns:")
# print(tennismylife.columns.tolist())

# # to inspect just the sackmann 2026 matches
# sackmann_2026 = sackmann[sackmann['source_year'] == 2026]
# print(sackmann_2026.head().to_string(index = False))

# # to find matches involving a particular player
# player = "Coco Gauff"

# coco_match = sackmann[
#     (sackmann['winner_name'] == player) | (sackmann['loser_name'] == player)]

# print(coco_match[
#     ['tourney_name', 'tourney_date', 'winner_name', 'loser_name', 'score', 'surface']
# ].head(10).to_string(index = False))

import pandas as pd

from environmental_data_pipeline import enrich_matches_with_environment
from step_02_load_and_compatre_providers import load_provider

sackmann = load_provider('sackmann', 'wta_matches_*.csv')
tennismylife = load_provider('tennismylife', '*_wta.csv')

KEY_COLUMNS = [
    "tourney_date",
    "winner_name",
    "loser_name",
    "score",
    "surface",
    "winner_rank",
    "loser_rank",
    "w_svpt",
    "l_svpt",
]
ENVIRONMENT_COLUMNS = [
    "venue_key",
    "weather_status",
    "weather_temperature_mean_f",
    "weather_precipitation_total_mm",
    "weather_wind_speed_max_kmh",
]

def quality_summary(df: pd.DataFrame, provider: str):
    dates = pd.to_datetime(
        df['tourney_date'].astype('string'),
        format = "%Y%m%d",
        errors = 'coerce'
    )

    scores = df['score'].fillna("").astype(str).str.strip().str.upper()

    non_completed = scores.str.contains(
        r"\b(?:RET|W/O|WALKOVER|DEF|ABN|ABD)\b",
        regex = True,
    )

    winner_rank = pd.to_numeric(df['winner_rank'], errors='coerce')
    loser_rank = pd.to_numeric(df['loser_rank'], errors='coerce')
    ranking_available = (winner_rank > 0) & (loser_rank > 0)

    winner_serve_points = pd.to_numeric(df['w_svpt'], errors='coerce')
    loser_serve_points = pd.to_numeric(df['l_svpt'], errors='coerce')
    serve_stats_available = (winner_serve_points > 0) & (loser_serve_points > 0)

    summary = pd.DataFrame([{
        'provider': provider,
        'rows': len(df),
        'columns': len(df.columns),
        'first_year': dates.dt.year.min(),
        'last_year': dates.dt.year.max(),
        'invalid_or_missing_dates': dates.isna().sum(),
        'missing_scores': scores.eq("").sum(),
        'non_completed_matches':non_completed.sum(),
        'duplicate_match_keys': df.duplicated(subset=['tourney_id', 'match_num']).sum(),
        'both_rankings_available_pct': round(ranking_available.mean() * 100, 2),
        'both_serve_states_available_pct': round(serve_stats_available.mean() * 100, 2),
    }])

    missingness = (
        df[KEY_COLUMNS]
        .isna()
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
        .rename('columns')
        .reset_index(name='missing_pct')
    )

    return summary, missingness


def environmental_quality_summary(df: pd.DataFrame, provider: str):
    """Report venue/weather coverage separately from raw source quality."""
    venue_mapped = df["venue_match_status"].eq("matched")
    weather_available = df["weather_status"].eq("ok")
    summary = pd.DataFrame([{
        "provider": provider,
        "weather_grain": "tournament_week",
        "venue_mapped_pct": round(venue_mapped.mean() * 100, 2),
        "tournament_week_weather_available_pct": round(
            weather_available.mean() * 100, 2
        ),
        "weather_rows_available": int(weather_available.sum()),
        "weather_rows_missing": int((~weather_available).sum()),
    }])
    missingness = (
        df[ENVIRONMENT_COLUMNS]
        .isna()
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
        .rename("column")
        .reset_index(name="missing_pct")
    )
    return summary, missingness


for provider_name, frame in {
    "Sackmann": sackmann,
    "TennisMyLife": tennismylife,
}.items():
    summary, missingness = quality_summary(frame, provider_name)
    environmental_frame = enrich_matches_with_environment(frame)
    environmental_summary, environmental_missingness = environmental_quality_summary(
        environmental_frame, provider_name
    )

    print(f"\n{'=' * 20} {provider_name} {'=' * 20}")
    print(summary.set_index('provider').T.to_string())

    print("\nMissingness in key columns (%):")
    print(missingness.to_string(index=False))

    print("\nEnvironmental enrichment coverage:")
    print(environmental_summary.set_index("provider").T.to_string())

    print("\nMissingness in environmental fields (%):")
    print(environmental_missingness.to_string(index=False))

