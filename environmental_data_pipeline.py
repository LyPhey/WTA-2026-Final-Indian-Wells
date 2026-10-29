#!/usr/bin/env python3
"""Create venue-aware, tournament-week weather enrichments for WTA match data.

The raw match files only contain a tournament-week start date, not an actual
match date or start time. This pipeline therefore produces *tournament-week*
weather aggregates. They are useful for exploratory analysis, but are not
match-day observations and must not be treated as pre-match features.

Workflow:
    python environmental_data_pipeline.py build-venue-template
    # Review data/reference/venue_reference_template.csv and add approved rows
    # to data/reference/venue_reference.csv.
    python environmental_data_pipeline.py fetch-weather
    python environmental_data_pipeline.py build-enriched-matches

Open-Meteo historical data are reanalysis data. Retain the attribution and
licensing information in DATA_PROVENANCE below with every downstream export.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import re
import time
from typing import Iterable

import pandas as pd
import requests


RAW = Path("data/raw")
REFERENCE_DIR = Path("data/reference")
PROCESSED_DIR = Path("data/processed")
VENUE_REFERENCE_PATH = REFERENCE_DIR / "venue_reference.csv"
VENUE_TEMPLATE_PATH = REFERENCE_DIR / "venue_reference_template.csv"
TOURNAMENT_WEEK_WEATHER_PATH = PROCESSED_DIR / "tournament_week_weather.csv"
ENRICHED_MATCHES_PATH = PROCESSED_DIR / "wta_matches_environment_enriched.csv"
LIVE_TENNIS_LOCATIONS_PATH = REFERENCE_DIR / "live_tennis_tournament_locations.geojson"
AUTO_GEOCODED_LOCATIONS_PATH = REFERENCE_DIR / "automatic_geocoded_locations.csv"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
LIVE_TENNIS_LOCATIONS_URL = "https://livetennisapi.com/data/tennis-tournaments.geojson"
DATA_PROVENANCE = (
    "Open-Meteo Historical Weather API; ERA5 reanalysis; CC BY 4.0; "
    "https://open-meteo.com/en/docs/historical-weather-api"
)
LIVE_TENNIS_PROVENANCE = (
    "Tournament locations: Live Tennis API (livetennisapi.com), city coordinates "
    "from GeoNames, CC BY 4.0"
)
WEATHER_GRAIN = "tournament_week"
DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "shortwave_radiation_sum",
    "sunshine_duration",
    "weather_code",
]
MATCH_SOURCES = {
    "Sackmann": ("sackmann", "wta_matches_*.csv"),
    "TennisMyLife": ("tennismylife", "*_wta.csv"),
}
# These aliases only resolve well-known tournament naming differences. They do
# not attempt to infer locations for team events or ambiguous tournament names.
LOCATION_NAME_ALIASES = {
    "ROLAND GARROS": "FRENCH OPEN",
}
GEOCODE_EXCLUDED_TOKENS = {
    "BJK",
    "CUP",
    "FED",
    "FINALS",
    "GROUP",
    "PLAYOFF",
    "QUALIFIER",
    "QUALIFYING",
    "RR",
    "TEAM",
    "VS",
}
REFERENCE_COLUMNS = [
    "venue_key",
    "tourney_code",
    "tourney_name_normalized",
    "valid_from",
    "valid_to",
    "venue_name",
    "city",
    "country",
    "latitude",
    "longitude",
    "elevation_m",
    "timezone",
    "indoor_override",
    "review_status",
    "reference_url",
    "notes",
]
ENVIRONMENT_COLUMNS = [
    "venue_match_status",
    "venue_key",
    "venue_name",
    "venue_city",
    "venue_country",
    "venue_latitude",
    "venue_longitude",
    "venue_elevation_m",
    "venue_timezone",
    "venue_indoor_override",
    "venue_mapping_method",
    "venue_location_precision",
    "weather_source",
    "weather_grain",
    "weather_status",
    "weather_week_start",
    "weather_week_end",
    "weather_days",
    "weather_temperature_mean_f",
    "weather_temperature_max_f",
    "weather_temperature_min_f",
    "weather_relative_humidity_mean_pct",
    "weather_precipitation_total_mm",
    "weather_wind_speed_max_kmh",
    "weather_wind_gusts_max_kmh",
    "weather_shortwave_radiation_total_mj_m2",
    "weather_sunshine_duration_hours",
    "weather_code_mode",
]


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize tournament names before matching sources to venue references."""
    return (
        series.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def source_year_from_filename(path: Path) -> int:
    match = re.search(r"(?:19|20)\d{2}", path.stem)
    if not match:
        raise ValueError(f"No year found in filename: {path.name}")
    return int(match.group())


def tournament_code(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"^\d{4}-", "", regex=True)


def tournament_week_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["tourney_date"].astype("string")
        + " | "
        + normalize_text(frame["tourney_name"])
    )


def read_all_source_matches() -> pd.DataFrame:
    """Read match files without importing later pipeline steps."""
    frames = []
    for display_name, (provider, pattern) in MATCH_SOURCES.items():
        for path in sorted((RAW / provider).glob(pattern)):
            frame = pd.read_csv(path, low_memory=False)
            frame["provider"] = provider
            frame["provider_label"] = display_name
            frame["source_file"] = path.name
            frame["source_year"] = source_year_from_filename(path)
            frames.append(frame)
    if not frames:
        raise ValueError("No provider match CSVs were found under data/raw.")
    return pd.concat(frames, ignore_index=True)


def source_indoor_values(series: pd.Series) -> str:
    values = sorted({str(value).strip() for value in series.dropna() if str(value).strip()})
    return ", ".join(values)


def build_venue_template(matches: pd.DataFrame) -> pd.DataFrame:
    """Create a reviewable venue-reference template from all raw tournaments."""
    template_input = matches.copy()
    template_input["tourney_code"] = tournament_code(template_input["tourney_id"])
    template_input["tourney_name_normalized"] = normalize_text(
        template_input["tourney_name"]
    )
    template_input["week_start"] = pd.to_datetime(
        template_input["tourney_date"].astype("string"), format="%Y%m%d", errors="coerce"
    )
    if "indoor" not in template_input:
        template_input["indoor"] = pd.NA

    template = (
        template_input.groupby(["tourney_code", "tourney_name_normalized"], dropna=False)
        .agg(
            sample_tourney_name=("tourney_name", "first"),
            first_seen=("week_start", "min"),
            last_seen=("week_start", "max"),
            source_providers=(
                "provider_label",
                lambda providers: ", ".join(sorted(set(providers))),
            ),
            source_indoor_values=("indoor", source_indoor_values),
            match_rows=("match_num", "size"),
        )
        .reset_index()
        .sort_values(["tourney_name_normalized", "first_seen"])
    )
    template["valid_from"] = template["first_seen"].dt.strftime("%Y-%m-%d")
    template["valid_to"] = template["last_seen"].dt.strftime("%Y-%m-%d")
    template["venue_key"] = ""
    template["venue_name"] = ""
    template["city"] = ""
    template["country"] = ""
    template["latitude"] = ""
    template["longitude"] = ""
    template["elevation_m"] = ""
    template["timezone"] = ""
    template["indoor_override"] = ""
    template["review_status"] = "needs_review"
    template["reference_url"] = ""
    template["notes"] = ""
    return template[
        REFERENCE_COLUMNS
        + [
            "sample_tourney_name",
            "first_seen",
            "last_seen",
            "source_providers",
            "source_indoor_values",
            "match_rows",
        ]
    ]


def write_venue_template(path: Path = VENUE_TEMPLATE_PATH) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    template = build_venue_template(read_all_source_matches())
    template.to_csv(path, index=False)
    return template


def read_venue_reference(path: Path = VENUE_REFERENCE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Venue reference not found: {path}. Start with {VENUE_TEMPLATE_PATH}."
        )
    reference = pd.read_csv(path, dtype="string")
    missing_columns = sorted(set(REFERENCE_COLUMNS) - set(reference.columns))
    if missing_columns:
        raise ValueError(f"Venue reference is missing columns: {missing_columns}")
    reference = reference[REFERENCE_COLUMNS].copy()
    reference["tourney_name_normalized"] = normalize_text(
        reference["tourney_name_normalized"]
    )
    reference["tourney_code"] = reference["tourney_code"].fillna("").str.strip()
    reference["valid_from"] = pd.to_datetime(reference["valid_from"], errors="coerce")
    reference["valid_to"] = pd.to_datetime(reference["valid_to"], errors="coerce")
    reference["valid_from"] = reference["valid_from"].fillna(pd.Timestamp("1900-01-01"))
    reference["valid_to"] = reference["valid_to"].fillna(pd.Timestamp("2100-12-31"))
    reference["latitude"] = pd.to_numeric(reference["latitude"], errors="coerce")
    reference["longitude"] = pd.to_numeric(reference["longitude"], errors="coerce")
    reference["elevation_m"] = pd.to_numeric(reference["elevation_m"], errors="coerce")
    reference["review_status"] = reference["review_status"].fillna("").str.lower().str.strip()
    return reference


def tournament_weeks(matches: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw provider rows to a unique tournament week for weather lookup."""
    weeks = matches.copy()
    weeks["week_start"] = pd.to_datetime(
        weeks["tourney_date"].astype("string"), format="%Y%m%d", errors="coerce"
    )
    weeks["tourney_name_normalized"] = normalize_text(weeks["tourney_name"])
    weeks["location_lookup_name"] = weeks["tourney_name_normalized"].replace(
        LOCATION_NAME_ALIASES
    )
    weeks["tournament_week_key"] = tournament_week_key(weeks)
    weeks["tourney_code"] = tournament_code(weeks["tourney_id"])
    if "indoor" not in weeks:
        weeks["indoor"] = pd.NA

    return (
        weeks.groupby("tournament_week_key", dropna=False)
        .agg(
            tourney_date=("tourney_date", "first"),
            tourney_name=("tourney_name", "first"),
            tourney_name_normalized=("tourney_name_normalized", "first"),
            location_lookup_name=("location_lookup_name", "first"),
            week_start=("week_start", "first"),
            tourney_codes=(
                "tourney_code",
                lambda values: " | ".join(sorted(set(values.astype(str)))),
            ),
            source_providers=(
                "provider_label",
                lambda providers: ", ".join(sorted(set(providers))),
            ),
            source_indoor_values=("indoor", source_indoor_values),
        )
        .reset_index()
    )


def sync_live_tennis_locations(
    path: Path = LIVE_TENNIS_LOCATIONS_PATH,
) -> pd.DataFrame:
    """Cache the published Live Tennis API tournament-city GeoJSON locally."""
    response = requests.get(LIVE_TENNIS_LOCATIONS_URL, timeout=(10, 60))
    response.raise_for_status()
    payload = response.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return load_live_tennis_locations(path)


def load_live_tennis_locations(
    path: Path = LIVE_TENNIS_LOCATIONS_PATH,
) -> pd.DataFrame:
    """Load unambiguous women's tournament-city locations from the cached GeoJSON."""
    columns = [
        "location_lookup_name",
        "venue_key",
        "venue_name",
        "city",
        "country",
        "latitude",
        "longitude",
        "elevation_m",
        "timezone",
        "indoor_override",
        "reference_url",
        "venue_mapping_method",
        "venue_location_precision",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [])
        if properties.get("gender") != "women" or len(coordinates) != 2:
            continue
        name = properties.get("name")
        if not name:
            continue
        records.append(
            {
                "location_lookup_name": normalize_text(pd.Series([name])).iloc[0],
                "venue_key": f"live-tennis-{normalize_text(pd.Series([name])).iloc[0].lower().replace(' ', '-')}",
                "venue_name": properties.get("city") or name,
                "city": properties.get("city"),
                "country": properties.get("country"),
                "latitude": coordinates[1],
                "longitude": coordinates[0],
                "elevation_m": pd.NA,
                "timezone": properties.get("time_zone"),
                "indoor_override": pd.NA,
                "reference_url": LIVE_TENNIS_LOCATIONS_URL,
                "venue_mapping_method": "published_city_location",
                "venue_location_precision": properties.get("precision", "city"),
            }
        )
    locations = pd.DataFrame(records, columns=columns)
    return locations.drop_duplicates(
        subset=["location_lookup_name", "latitude", "longitude"], keep="first"
    )


def location_columns() -> list[str]:
    return [
        "location_lookup_name",
        "venue_key",
        "venue_name",
        "city",
        "country",
        "latitude",
        "longitude",
        "elevation_m",
        "timezone",
        "indoor_override",
        "reference_url",
        "venue_mapping_method",
        "venue_location_precision",
    ]


def is_geocode_candidate(location_name: str) -> bool:
    tokens = set(location_name.split())
    return bool(location_name) and not bool(tokens & GEOCODE_EXCLUDED_TOKENS)


def load_automatic_geocoded_locations(
    path: Path = AUTO_GEOCODED_LOCATIONS_PATH,
) -> pd.DataFrame:
    columns = [*location_columns(), "geocode_status", "geocode_error"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    locations = pd.read_csv(path, low_memory=False)
    for column in columns:
        if column not in locations:
            locations[column] = pd.NA
    return locations[columns]


def sync_automatic_geocoded_locations(
    venues: pd.DataFrame,
    refresh: bool = False,
    path: Path = AUTO_GEOCODED_LOCATIONS_PATH,
) -> pd.DataFrame:
    """Cache high-confidence city geocodes for otherwise unmapped tournaments.

    This only accepts a result when the returned place name exactly matches the
    normalized tournament name. Team events and descriptive tournament names
    are skipped rather than assigned an invented location.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cached = load_automatic_geocoded_locations(path)
    attempted = set(cached["location_lookup_name"].dropna()) if not refresh else set()
    candidates = (
        venues.loc[venues["venue_match_status"].eq("unmapped")]
        .groupby("location_lookup_name", as_index=False)
        .agg(tourney_name=("tourney_name", "first"))
    )
    records = []
    session = requests.Session()
    session.headers.update({"User-Agent": "WTA-ML-environmental-enrichment/1.0"})
    for _, candidate in candidates.iterrows():
        location_name = candidate["location_lookup_name"]
        if location_name in attempted:
            continue
        base = {
            "location_lookup_name": location_name,
            "venue_key": pd.NA,
            "venue_name": pd.NA,
            "city": pd.NA,
            "country": pd.NA,
            "latitude": pd.NA,
            "longitude": pd.NA,
            "elevation_m": pd.NA,
            "timezone": pd.NA,
            "indoor_override": pd.NA,
            "reference_url": OPEN_METEO_GEOCODING_URL,
            "venue_mapping_method": "automatic_name_geocode",
            "venue_location_precision": "city",
        }
        if not is_geocode_candidate(location_name):
            records.append(
                {
                    **base,
                    "geocode_status": "skipped_ambiguous_name",
                    "geocode_error": "Tournament name does not safely identify one city.",
                }
            )
            continue
        try:
            response = session.get(
                OPEN_METEO_GEOCODING_URL,
                params={"name": candidate["tourney_name"], "count": 10, "language": "en"},
                timeout=(10, 60),
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            exact = [
                result
                for result in results
                if normalize_text(pd.Series([result.get("name", "")])).iloc[0]
                == location_name
            ]
            if len(exact) != 1:
                records.append(
                    {
                        **base,
                        "geocode_status": "unresolved",
                        "geocode_error": "No single exact city-name result.",
                    }
                )
                continue
            result = exact[0]
            records.append(
                {
                    **base,
                    "venue_key": f"open-meteo-{location_name.lower().replace(' ', '-')}",
                    "venue_name": result.get("name"),
                    "city": result.get("name"),
                    "country": result.get("country"),
                    "latitude": result.get("latitude"),
                    "longitude": result.get("longitude"),
                    "elevation_m": result.get("elevation"),
                    "timezone": result.get("timezone"),
                    "geocode_status": "ok",
                    "geocode_error": "",
                }
            )
        except requests.RequestException as error:
            records.append(
                {**base, "geocode_status": "error", "geocode_error": str(error)}
            )
        time.sleep(0.05)

    refreshed = pd.DataFrame(records)
    if refresh and not refreshed.empty:
        cached = cached[
            ~cached["location_lookup_name"].isin(refreshed["location_lookup_name"])
        ]
    result = pd.concat([cached, refreshed], ignore_index=True, sort=False)
    if not result.empty:
        result = result.drop_duplicates("location_lookup_name", keep="last")
        result.to_csv(path, index=False)
    return result


def apply_automatic_locations(
    venues: pd.DataFrame, locations: pd.DataFrame
) -> pd.DataFrame:
    """Fill still-unmapped weeks from a one-to-one automatic city location."""
    if locations.empty:
        return venues
    valid_locations = locations[locations["geocode_status"].eq("ok")].copy()
    if valid_locations.empty:
        return venues
    valid_locations = valid_locations.drop_duplicates("location_lookup_name", keep=False)
    candidates = venues.loc[
        venues["venue_match_status"].eq("unmapped"),
        ["tournament_week_key", "location_lookup_name"],
    ].merge(valid_locations, on="location_lookup_name", how="left")
    selected = candidates[candidates["venue_key"].notna()].set_index("tournament_week_key")
    for column in location_columns()[1:]:
        replacement = venues["tournament_week_key"].map(selected[column])
        venues[column] = venues[column].fillna(replacement)
    auto_matched = venues["tournament_week_key"].isin(selected.index)
    venues.loc[auto_matched, "venue_match_status"] = "matched"
    return venues


def match_venue_references(
    weeks: pd.DataFrame,
    reference: pd.DataFrame,
    live_tennis_locations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach a manual venue first, then published unambiguous city locations."""
    approved = reference[
        reference["review_status"].eq("approved")
        & reference["latitude"].notna()
        & reference["longitude"].notna()
    ].copy().rename(columns={"tourney_code": "reference_tourney_code"})
    if approved.empty:
        candidates = pd.DataFrame()
        match_count = pd.Series(dtype="int64", name="venue_candidate_count")
        selected = pd.DataFrame()
    else:
        candidates = weeks.merge(
            approved,
            on="tourney_name_normalized",
            how="left",
            suffixes=("", "_reference"),
        )
        code_matches = candidates.apply(
            lambda row: pd.isna(row["reference_tourney_code"])
            or not str(row["reference_tourney_code"]).strip()
            or str(row["reference_tourney_code"]) in row["tourney_codes"].split(" | "),
            axis=1,
        )
        date_matches = candidates["week_start"].between(
            candidates["valid_from"], candidates["valid_to"]
        )
        candidates = candidates[
            code_matches & date_matches & candidates["venue_key"].notna()
        ]
        match_count = candidates.groupby("tournament_week_key").size().rename(
            "venue_candidate_count"
        )
        selected = candidates.join(match_count, on="tournament_week_key")
        selected = selected[selected["venue_candidate_count"].eq(1)].drop_duplicates(
            "tournament_week_key"
        )
    venue_columns = [
        "tournament_week_key",
        "venue_key",
        "venue_name",
        "city",
        "country",
        "latitude",
        "longitude",
        "elevation_m",
        "timezone",
        "indoor_override",
        "reference_url",
    ]
    if selected.empty:
        result = weeks.copy()
        for column in venue_columns[1:]:
            result[column] = pd.NA
    else:
        result = weeks.merge(selected[venue_columns], on="tournament_week_key", how="left")
    result = result.join(match_count, on="tournament_week_key")
    result["venue_match_status"] = "unmapped"
    result.loc[result["venue_candidate_count"].gt(1), "venue_match_status"] = "ambiguous"
    result.loc[result["venue_key"].notna(), "venue_match_status"] = "matched"
    result["venue_mapping_method"] = pd.NA
    result["venue_location_precision"] = pd.NA
    result.loc[result["venue_key"].notna(), "venue_mapping_method"] = "manual_reference"
    result.loc[result["venue_key"].notna(), "venue_location_precision"] = "venue"

    if live_tennis_locations is None or live_tennis_locations.empty:
        return result

    auto_candidates = result.loc[
        result["venue_key"].isna(), ["tournament_week_key", "location_lookup_name"]
    ].merge(live_tennis_locations, on="location_lookup_name", how="left")
    auto_counts = auto_candidates.groupby("tournament_week_key")["venue_key"].count()
    auto_selected = auto_candidates.join(
        auto_counts.rename("auto_candidate_count"), on="tournament_week_key"
    )
    auto_selected = auto_selected[auto_selected["auto_candidate_count"].eq(1)].set_index(
        "tournament_week_key"
    )
    for column in venue_columns[1:] + ["venue_mapping_method", "venue_location_precision"]:
        replacement = result["tournament_week_key"].map(auto_selected[column])
        result[column] = result[column].fillna(replacement)
    auto_matched = result["tournament_week_key"].isin(auto_selected.index)
    result.loc[auto_matched, "venue_match_status"] = "matched"
    return result


def numeric_mean(values: Iterable[object]) -> float | None:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return None if series.empty else float(series.mean())


def numeric_max(values: Iterable[object]) -> float | None:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return None if series.empty else float(series.max())


def numeric_sum(values: Iterable[object]) -> float | None:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return None if series.empty else float(series.sum())


def weather_code_mode(values: Iterable[object]) -> int | None:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if series.empty:
        return None
    return int(series.mode().iloc[0])


def fetch_week_weather(
    venues: pd.DataFrame,
    refresh: bool = False,
    output_path: Path = TOURNAMENT_WEEK_WEATHER_PATH,
) -> pd.DataFrame:
    """Fetch and cache one daily-weather aggregate for each approved venue week."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        cached = pd.read_csv(output_path, low_memory=False)
    else:
        cached = pd.DataFrame()
    if not refresh and {
        "tournament_week_key",
        "weather_status",
        "weather_temperature_mean_f",
    }.issubset(cached.columns):
        cached_keys = set(
            cached.loc[
                cached["weather_status"].eq("ok")
                & cached["weather_temperature_mean_f"].notna(),
                "tournament_week_key",
            ]
        )
    else:
        cached_keys = set()

    today = date.today()
    records = []
    session = requests.Session()
    session.headers.update({"User-Agent": "WTA-ML-environmental-enrichment/1.0"})
    tasks = venues[venues["venue_match_status"].eq("matched")].copy()
    for _, venue in tasks.iterrows():
        key = venue["tournament_week_key"]
        if key in cached_keys:
            continue
        week_start = pd.Timestamp(venue["week_start"]).date()
        week_end = week_start + timedelta(days=6)
        base = {
            "tournament_week_key": key,
            "tourney_date": venue["tourney_date"],
            "tourney_name": venue["tourney_name"],
            "venue_match_status": venue["venue_match_status"],
            "venue_key": venue["venue_key"],
            "venue_name": venue["venue_name"],
            "venue_city": venue["city"],
            "venue_country": venue["country"],
            "venue_latitude": venue["latitude"],
            "venue_longitude": venue["longitude"],
            "venue_elevation_m": venue["elevation_m"],
            "venue_timezone": venue["timezone"],
            "venue_indoor_override": venue["indoor_override"],
            "venue_mapping_method": venue["venue_mapping_method"],
            "venue_location_precision": venue["venue_location_precision"],
            "weather_source": DATA_PROVENANCE,
            "weather_grain": WEATHER_GRAIN,
            "weather_model": "era5",
            "weather_week_start": week_start.isoformat(),
            "weather_week_end": week_end.isoformat(),
        }
        if week_end > today:
            records.append(
                {
                    **base,
                    "weather_status": "skipped_future_week",
                    "weather_error": "Tournament week has not finished.",
                }
            )
            continue
        try:
            response = session.get(
                OPEN_METEO_ARCHIVE_URL,
                params={
                    "latitude": venue["latitude"],
                    "longitude": venue["longitude"],
                    "start_date": week_start.isoformat(),
                    "end_date": week_end.isoformat(),
                    "daily": ",".join(DAILY_VARIABLES),
                    "timezone": "auto",
                    "models": "era5",
                    "temperature_unit": "fahrenheit",
                },
                timeout=(10, 60),
            )
            response.raise_for_status()
            daily = response.json().get("daily", {})
            if not daily.get("time"):
                raise ValueError("The API response did not include daily weather data.")
            records.append(
                {
                    **base,
                    "venue_elevation_m": response.json().get(
                        "elevation", venue["elevation_m"]
                    ),
                    "weather_status": "ok",
                    "weather_error": "",
                    "weather_days": len(daily["time"]),
                    "weather_temperature_mean_f": numeric_mean(
                        daily.get("temperature_2m_mean", [])
                    ),
                    "weather_temperature_max_f": numeric_max(
                        daily.get("temperature_2m_max", [])
                    ),
                    "weather_temperature_min_f": numeric_min(
                        daily.get("temperature_2m_min", [])
                    ),
                    "weather_relative_humidity_mean_pct": numeric_mean(
                        daily.get("relative_humidity_2m_mean", [])
                    ),
                    "weather_precipitation_total_mm": numeric_sum(
                        daily.get("precipitation_sum", [])
                    ),
                    "weather_wind_speed_max_kmh": numeric_max(
                        daily.get("wind_speed_10m_max", [])
                    ),
                    "weather_wind_gusts_max_kmh": numeric_max(
                        daily.get("wind_gusts_10m_max", [])
                    ),
                    "weather_shortwave_radiation_total_mj_m2": numeric_sum(
                        daily.get("shortwave_radiation_sum", [])
                    ),
                    "weather_sunshine_duration_hours": (
                        numeric_sum(daily.get("sunshine_duration", [])) or 0
                    )
                    / 3600,
                    "weather_code_mode": weather_code_mode(daily.get("weather_code", [])),
                }
            )
        except (requests.RequestException, ValueError, TypeError) as error:
            records.append(
                {**base, "weather_status": "error", "weather_error": str(error)}
            )
        time.sleep(0.1)

    fetched = pd.DataFrame(records)
    if refresh and not cached.empty and not fetched.empty:
        cached = cached[~cached["tournament_week_key"].isin(fetched["tournament_week_key"])]
    result = pd.concat([cached, fetched], ignore_index=True, sort=False)
    if not result.empty:
        result = result.drop_duplicates("tournament_week_key", keep="last")
        result = result.drop(
            columns=[
                "weather_temperature_mean_c",
                "weather_temperature_max_c",
                "weather_temperature_min_c",
            ],
            errors="ignore",
        )
        result.to_csv(output_path, index=False)
    return result


def numeric_min(values: Iterable[object]) -> float | None:
    series = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    return None if series.empty else float(series.min())


def load_tournament_week_weather(
    path: Path = TOURNAMENT_WEEK_WEATHER_PATH,
) -> pd.DataFrame:
    """Load the cached tournament-week weather table, or a safe empty table."""
    if not path.exists():
        return pd.DataFrame(columns=["tournament_week_key", *ENVIRONMENT_COLUMNS])
    weather = pd.read_csv(path, low_memory=False)
    if weather["tournament_week_key"].duplicated().any():
        raise ValueError(f"Duplicate tournament-week weather records found in {path}.")
    return weather


def enrich_matches_with_environment(matches: pd.DataFrame) -> pd.DataFrame:
    """Left-join cached weekly weather without dropping or duplicating match rows."""
    enriched = matches.copy()
    enriched["tournament_week_key"] = tournament_week_key(enriched)
    weather = load_tournament_week_weather()
    if weather.empty:
        for column in ENVIRONMENT_COLUMNS:
            if column not in enriched:
                enriched[column] = pd.NA
    else:
        weather_columns = [
            column
            for column in ["tournament_week_key", *ENVIRONMENT_COLUMNS]
            if column in weather.columns
        ]
        enriched = enriched.merge(
            weather[weather_columns],
            on="tournament_week_key",
            how="left",
            validate="many_to_one",
        )
    for column in ENVIRONMENT_COLUMNS:
        if column not in enriched:
            enriched[column] = pd.NA
    enriched["venue_match_status"] = enriched["venue_match_status"].fillna("unmapped")
    enriched["weather_status"] = enriched["weather_status"].fillna(
        "unavailable_no_mapped_location"
    )
    enriched["weather_grain"] = enriched["weather_grain"].fillna(WEATHER_GRAIN)
    return enriched


def build_enriched_matches(
    output_path: Path = ENRICHED_MATCHES_PATH,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched = enrich_matches_with_environment(read_all_source_matches())
    enriched.to_csv(output_path, index=False)
    return enriched


def environmental_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize which source rows have a mapped venue and cached weekly weather."""
    provider_column = "provider_label" if "provider_label" in frame else "provider"
    summary = (
        frame.assign(
            venue_mapped=frame["venue_match_status"].eq("matched"),
            weather_available=frame["weather_status"].eq("ok"),
        )
        .groupby(provider_column, dropna=False)
        .agg(
            rows=("provider", "size"),
            venue_mapped=("venue_mapped", "sum"),
            weather_available=("weather_available", "sum"),
        )
        .reset_index()
        .assign(
            venue_mapped_pct=lambda x: (100 * x["venue_mapped"] / x["rows"]).round(2),
            weather_available_pct=lambda x: (
                100 * x["weather_available"] / x["rows"]
            ).round(2),
        )
    )
    return summary.rename(columns={provider_column: "provider_label"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build-venue-template", help="Create a venue mapping review template.")
    subparsers.add_parser(
        "sync-tournament-locations",
        help="Download the published Live Tennis API women's tournament-city map.",
    )
    geocode_parser = subparsers.add_parser(
        "geocode-unmapped",
        help="Cache exact-name city geocodes for unmapped tournament names.",
    )
    geocode_parser.add_argument(
        "--refresh", action="store_true", help="Retry previously cached geocoding results."
    )
    fetch_parser = subparsers.add_parser(
        "fetch-weather", help="Fetch cached weekly weather for approved venue mappings."
    )
    fetch_parser.add_argument("--refresh", action="store_true", help="Refresh existing weeks.")
    subparsers.add_parser(
        "build-enriched-matches", help="Write source-level matches joined to cached weather."
    )
    run_parser = subparsers.add_parser("run", help="Run the complete safe enrichment workflow.")
    run_parser.add_argument("--refresh", action="store_true", help="Refresh existing weather weeks.")
    args = parser.parse_args()

    if args.command in {"build-venue-template", "run"}:
        template = write_venue_template()
        print(f"Wrote {VENUE_TEMPLATE_PATH} with {len(template):,} venue-mapping candidates.")

    live_locations = pd.DataFrame()
    if args.command in {
        "sync-tournament-locations",
        "geocode-unmapped",
        "fetch-weather",
        "run",
    }:
        live_locations = sync_live_tennis_locations()
        print(
            f"Cached {len(live_locations):,} published women's tournament-city locations "
            f"at {LIVE_TENNIS_LOCATIONS_PATH}."
        )

    venues = pd.DataFrame()
    automatic_locations = pd.DataFrame()
    if args.command in {"geocode-unmapped", "fetch-weather", "run"}:
        matches = read_all_source_matches()
        venues = match_venue_references(
            tournament_weeks(matches), read_venue_reference(), live_locations
        )
        automatic_locations = sync_automatic_geocoded_locations(
            venues, refresh=getattr(args, "refresh", False)
        )
        print(
            f"Cached {len(automatic_locations):,} automatic geocoding candidates at "
            f"{AUTO_GEOCODED_LOCATIONS_PATH}; review them before promoting any to "
            "venue_reference.csv."
        )

    if args.command in {"fetch-weather", "run"}:
        weather = fetch_week_weather(venues, refresh=args.refresh)
        mapped = int(venues["venue_match_status"].eq("matched").sum())
        print(f"Matched {mapped:,}/{len(venues):,} tournament weeks to approved venues.")
        print(f"Wrote {TOURNAMENT_WEEK_WEATHER_PATH} with {len(weather):,} cached results.")

    if args.command in {"build-enriched-matches", "run"}:
        enriched = build_enriched_matches()
        coverage = environmental_coverage(enriched)
        print(f"Wrote {ENRICHED_MATCHES_PATH} with {len(enriched):,} source-level match rows.")
        print(coverage.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
