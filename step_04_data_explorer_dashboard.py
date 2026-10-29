"""Interactive source-quality explorer for the WTA match datasets.

Run from this project directory:
    streamlit run step_04_data_explorer_dashboard.py
"""

from __future__ import annotations

from datetime import date
from io import BytesIO
import re
import unicodedata
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
import plotly.express as px
import requests
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
WTA_API_BASE = "https://api.wtatennis.com"
WTA_PHOTO_QUERY_LIMIT = 20
WTA_RESOURCE_VERSION = "v7.53.16"
HAND_LABELS = {"R": "Right-handed", "L": "Left-handed"}
MCP_WTA_OVERVIEW_URL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/"
    "master/charting-w-stats-Overview.csv"
)
MCP_WTA_MATCHES_URL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/"
    "master/charting-w-matches.csv"
)
MCP_WTA_SHOT_DIRECTION_URL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/"
    "master/charting-w-stats-ShotDirection.csv"
)
MCP_WTA_SHOT_TYPES_URL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/"
    "master/charting-w-stats-ShotTypes.csv"
)
MCP_WTA_RALLY_URL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/"
    "master/charting-w-stats-Rally.csv"
)
MCP_SHOT_WINNER_COLUMNS = [
    "match_id",
    "source_year",
    "tourney_key",
    "player_key",
    "opponent_key",
    "winners_fh",
    "winners_bh",
]
MCP_SHOT_DIRECTION_COLUMNS = [
    "match_id",
    "source_year",
    "tourney_key",
    "player_key",
    "opponent_key",
    "row",
    "crosscourt",
    "down_middle",
    "down_the_line",
    "inside_out",
    "inside_in",
]
MCP_SHOT_TYPE_COLUMNS = [
    "match_id",
    "source_year",
    "tourney_key",
    "player_key",
    "opponent_key",
    "row",
    "shots",
    "winners",
    "unforced",
]
MCP_RALLY_COLUMNS = [
    "match_id",
    "source_year",
    "tourney_key",
    "player_key",
    "opponent_key",
    "rally_band",
    "points",
    "points_won",
]


def normalized_player_name(name: object) -> str:
    """Normalize a label before matching it to an official WTA player record."""
    if not isinstance(name, str):
        if pd.isna(name):
            return ""
        name = str(name)
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_name.lower()).strip()


def resized_wta_photo_url(url: str, *, width: int = 260, height: int = 347) -> str:
    """Request a display-sized derivative from WTA's official photo service."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update({"width": str(width), "height": str(height)})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def wta_flag_url(country_code: str) -> str | None:
    """Return the WTA-hosted flag asset for a three-letter country code."""
    code = country_code.upper().strip()
    if not re.fullmatch(r"[A-Z]{3}", code):
        return None
    return (
        "https://www.wtatennis.com/"
        f"resources/{WTA_RESOURCE_VERSION}/i/elements/flags/{code.lower()}.svg"
    )


def age_from_date_of_birth(date_of_birth: str | None) -> int | None:
    """Return a player's current age from WTA's ISO date-of-birth value."""
    if not date_of_birth:
        return None
    try:
        birthday = date.fromisoformat(date_of_birth)
    except ValueError:
        return None
    today = date.today()
    return today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))


def normalized_hand(value: object) -> str | None:
    """Normalize WTA's descriptive and raw-provider hand values to R or L."""
    text = str(value).strip().upper()
    if text in {"R", "RIGHT", "RIGHT-HANDED"} or text.startswith("RIGHT"):
        return "R"
    if text in {"L", "LEFT", "LEFT-HANDED"} or text.startswith("LEFT"):
        return "L"
    return None


def hand_label(hand: str | None) -> str:
    """Return a reader-friendly label for a normalized hand value."""
    return HAND_LABELS.get(hand or "", "Not available")


def empty_mcp_shot_winner_frame() -> pd.DataFrame:
    """Return an empty, schema-consistent charted-shot statistics frame."""
    return pd.DataFrame(columns=MCP_SHOT_WINNER_COLUMNS)


@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner=False)
def load_mcp_wta_shot_winner_stats() -> pd.DataFrame:
    """Load forehand/backhand winners from the Match Charting Project.

    This is a crowdsourced, charted-match sample rather than a complete WTA
    results feed. Its CC BY-NC-SA 4.0 terms require attribution and
    noncommercial use. Only Total match rows are retained.
    """
    try:
        overview_response = requests.get(MCP_WTA_OVERVIEW_URL, timeout=30)
        overview_response.raise_for_status()
        matches_response = requests.get(MCP_WTA_MATCHES_URL, timeout=30)
        matches_response.raise_for_status()
        overview = pd.read_csv(BytesIO(overview_response.content))
        charted_matches = pd.read_csv(BytesIO(matches_response.content))
    except (requests.RequestException, pd.errors.ParserError, UnicodeDecodeError):
        return empty_mcp_shot_winner_frame()

    required_overview = {"match_id", "player", "set", "winners_fh", "winners_bh"}
    required_matches = {"match_id", "Tournament"}
    if not required_overview.issubset(overview) or not required_matches.issubset(
        charted_matches
    ):
        return empty_mcp_shot_winner_frame()

    shot_winners = overview.loc[overview["set"].eq("Total")].copy()
    shot_winners["source_year"] = pd.to_numeric(
        shot_winners["match_id"].astype(str).str[:4], errors="coerce"
    ).astype("Int64")
    shot_winners["player_key"] = shot_winners["player"].map(normalized_player_name)

    players_per_match = shot_winners.groupby("match_id")["player"].agg(list)

    def opponent_in_charted_match(row: pd.Series) -> str | None:
        opponents = [
            player for player in players_per_match[row["match_id"]] if player != row["player"]
        ]
        return opponents[0] if len(opponents) == 1 else None

    shot_winners["opponent"] = shot_winners.apply(opponent_in_charted_match, axis=1)
    shot_winners["opponent_key"] = shot_winners["opponent"].map(normalized_player_name)
    tournament_names = (
        charted_matches[["match_id", "Tournament"]]
        .dropna(subset=["Tournament"])
        .drop_duplicates("match_id")
        .rename(columns={"Tournament": "mcp_tourney_name"})
    )
    shot_winners = shot_winners.merge(
        tournament_names, on="match_id", how="left", validate="many_to_one"
    )
    shot_winners["tourney_key"] = shot_winners["mcp_tourney_name"].map(
        normalized_player_name
    )
    shot_winners["winners_fh"] = pd.to_numeric(
        shot_winners["winners_fh"], errors="coerce"
    )
    shot_winners["winners_bh"] = pd.to_numeric(
        shot_winners["winners_bh"], errors="coerce"
    )
    shot_winners = shot_winners.loc[:, MCP_SHOT_WINNER_COLUMNS].dropna(
        subset=["source_year", "tourney_key", "player_key", "opponent_key"]
    )
    return shot_winners.loc[
        shot_winners[["tourney_key", "player_key", "opponent_key"]].ne("").all(axis=1)
    ]


def matched_shot_winner_stats(log: pd.DataFrame, player: str) -> pd.DataFrame:
    """Return only unambiguous charted matches from the filtered player log."""
    charted = load_mcp_wta_shot_winner_stats()
    if charted.empty:
        return empty_mcp_shot_winner_frame()

    join_columns = ["source_year", "tourney_key", "player_key", "opponent_key"]
    source_keys = log[["source_year", "tourney_name", "opponent"]].copy()
    source_keys["source_year"] = pd.to_numeric(
        source_keys["source_year"], errors="coerce"
    ).astype("Int64")
    source_keys["tourney_key"] = source_keys.pop("tourney_name").map(
        normalized_player_name
    )
    source_keys["player_key"] = normalized_player_name(player)
    source_keys["opponent_key"] = source_keys.pop("opponent").map(
        normalized_player_name
    )

    charted = charted.loc[
        charted["player_key"].eq(normalized_player_name(player))
    ].copy()
    source_counts = source_keys.value_counts(join_columns)
    charted_counts = charted.value_counts(join_columns)
    unique_source_keys = source_counts.loc[source_counts.eq(1)].index
    unique_charted_keys = charted_counts.loc[charted_counts.eq(1)].index
    matching_keys = unique_source_keys.intersection(unique_charted_keys)
    if matching_keys.empty:
        return empty_mcp_shot_winner_frame()

    return charted.set_index(join_columns).loc[
        lambda indexed: indexed.index.isin(matching_keys)
    ].reset_index()


def empty_mcp_frame(columns: list[str]) -> pd.DataFrame:
    """Return an empty frame with the requested Match Charting Project schema."""
    return pd.DataFrame(columns=columns)


def mcp_player_match_index(charted_matches: pd.DataFrame) -> pd.DataFrame:
    """Return one player-relative metadata row for each charted WTA match."""
    required_columns = {"match_id", "Player 1", "Player 2", "Date", "Tournament"}
    if not required_columns.issubset(charted_matches):
        return pd.DataFrame(
            columns=[
                "match_id",
                "player",
                "source_year",
                "tourney_key",
                "player_key",
                "opponent_key",
            ]
        )

    match_metadata = charted_matches.loc[:, list(required_columns)].drop_duplicates(
        "match_id"
    )
    match_metadata["source_year"] = pd.to_datetime(
        match_metadata["Date"].astype("string"), format="%Y%m%d", errors="coerce"
    ).dt.year.astype("Int64")
    first_player = match_metadata.rename(
        columns={"Player 1": "player", "Player 2": "opponent"}
    )
    second_player = match_metadata.rename(
        columns={"Player 2": "player", "Player 1": "opponent"}
    )
    index = pd.concat([first_player, second_player], ignore_index=True)
    index["tourney_key"] = index["Tournament"].map(normalized_player_name)
    index["player_key"] = index["player"].map(normalized_player_name)
    index["opponent_key"] = index["opponent"].map(normalized_player_name)
    return index.loc[
        :,
        [
            "match_id",
            "player",
            "source_year",
            "tourney_key",
            "player_key",
            "opponent_key",
        ],
    ].dropna(subset=["source_year"])


def matched_mcp_chart_rows(
    log: pd.DataFrame, player: str, charted: pd.DataFrame
) -> pd.DataFrame:
    """Return all charted rows whose underlying match links uniquely to the player log."""
    if charted.empty:
        return charted.copy()

    join_columns = ["source_year", "tourney_key", "player_key", "opponent_key"]
    source_keys = log[["source_year", "tourney_name", "opponent"]].copy()
    source_keys["source_year"] = pd.to_numeric(
        source_keys["source_year"], errors="coerce"
    ).astype("Int64")
    source_keys["tourney_key"] = source_keys.pop("tourney_name").map(
        normalized_player_name
    )
    source_keys["player_key"] = normalized_player_name(player)
    source_keys["opponent_key"] = source_keys.pop("opponent").map(
        normalized_player_name
    )
    charted = charted.loc[
        charted["player_key"].eq(normalized_player_name(player))
    ].copy()
    source_counts = source_keys.value_counts(join_columns)
    charted_match_keys = charted.loc[:, join_columns].drop_duplicates()
    charted_counts = charted_match_keys.value_counts(join_columns)
    matching_keys = source_counts.loc[source_counts.eq(1)].index.intersection(
        charted_counts.loc[charted_counts.eq(1)].index
    )
    if matching_keys.empty:
        return charted.iloc[0:0].copy()
    return charted.set_index(join_columns).loc[
        lambda indexed: indexed.index.isin(matching_keys)
    ].reset_index()


@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner=False)
def load_mcp_wta_shot_direction_stats() -> pd.DataFrame:
    """Load charted groundstroke-direction counts with player-relative metadata."""
    try:
        direction_response = requests.get(MCP_WTA_SHOT_DIRECTION_URL, timeout=30)
        direction_response.raise_for_status()
        matches_response = requests.get(MCP_WTA_MATCHES_URL, timeout=30)
        matches_response.raise_for_status()
        directions = pd.read_csv(BytesIO(direction_response.content))
        charted_matches = pd.read_csv(BytesIO(matches_response.content))
    except (requests.RequestException, pd.errors.ParserError, UnicodeDecodeError):
        return empty_mcp_frame(MCP_SHOT_DIRECTION_COLUMNS)

    direction_counts = [
        "crosscourt",
        "down_middle",
        "down_the_line",
        "inside_out",
        "inside_in",
    ]
    required_columns = {"match_id", "player", "row", *direction_counts}
    if not required_columns.issubset(directions):
        return empty_mcp_frame(MCP_SHOT_DIRECTION_COLUMNS)
    directions = directions.loc[directions["row"].isin(["F", "B"])].copy()
    for column in direction_counts:
        directions[column] = pd.to_numeric(directions[column], errors="coerce")
    indexed = directions.merge(
        mcp_player_match_index(charted_matches),
        on=["match_id", "player"],
        how="inner",
        validate="many_to_one",
    )
    return indexed.loc[:, MCP_SHOT_DIRECTION_COLUMNS]


@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner=False)
def load_mcp_wta_shot_type_stats() -> pd.DataFrame:
    """Load charted forehand, backhand, and net shot totals with outcomes."""
    try:
        types_response = requests.get(MCP_WTA_SHOT_TYPES_URL, timeout=30)
        types_response.raise_for_status()
        matches_response = requests.get(MCP_WTA_MATCHES_URL, timeout=30)
        matches_response.raise_for_status()
        shot_types = pd.read_csv(BytesIO(types_response.content))
        charted_matches = pd.read_csv(BytesIO(matches_response.content))
    except (requests.RequestException, pd.errors.ParserError, UnicodeDecodeError):
        return empty_mcp_frame(MCP_SHOT_TYPE_COLUMNS)

    required_columns = {"match_id", "player", "row", "shots", "winners", "unforced"}
    if not required_columns.issubset(shot_types):
        return empty_mcp_frame(MCP_SHOT_TYPE_COLUMNS)
    shot_types = shot_types.loc[shot_types["row"].isin(["Fgs", "Bgs", "Net"])].copy()
    for column in ("shots", "winners", "unforced"):
        shot_types[column] = pd.to_numeric(shot_types[column], errors="coerce")
    indexed = shot_types.merge(
        mcp_player_match_index(charted_matches),
        on=["match_id", "player"],
        how="inner",
        validate="many_to_one",
    )
    return indexed.loc[:, MCP_SHOT_TYPE_COLUMNS]


@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner=False)
def load_mcp_wta_rally_stats() -> pd.DataFrame:
    """Load charted player-relative points and wins by rally-length band."""
    try:
        rally_response = requests.get(MCP_WTA_RALLY_URL, timeout=30)
        rally_response.raise_for_status()
        matches_response = requests.get(MCP_WTA_MATCHES_URL, timeout=30)
        matches_response.raise_for_status()
        rallies = pd.read_csv(BytesIO(rally_response.content))
        charted_matches = pd.read_csv(BytesIO(matches_response.content))
    except (requests.RequestException, pd.errors.ParserError, UnicodeDecodeError):
        return empty_mcp_frame(MCP_RALLY_COLUMNS)

    required_columns = {
        "match_id",
        "server",
        "returner",
        "row",
        "pts",
        "pl1_won",
        "pl2_won",
    }
    rally_bands = ["1-3", "4-6", "7-9", "10"]
    if not required_columns.issubset(rallies):
        return empty_mcp_frame(MCP_RALLY_COLUMNS)
    rallies = rallies.loc[rallies["row"].isin(rally_bands)].copy()
    server_rows = rallies.rename(
        columns={"server": "player", "returner": "opponent", "row": "rally_band", "pts": "points", "pl1_won": "points_won"}
    )
    returner_rows = rallies.rename(
        columns={"returner": "player", "server": "opponent", "row": "rally_band", "pts": "points", "pl2_won": "points_won"}
    )
    player_rallies = pd.concat(
        [
            server_rows[["match_id", "player", "rally_band", "points", "points_won"]],
            returner_rows[["match_id", "player", "rally_band", "points", "points_won"]],
        ],
        ignore_index=True,
    )
    for column in ("points", "points_won"):
        player_rallies[column] = pd.to_numeric(player_rallies[column], errors="coerce")
    indexed = player_rallies.merge(
        mcp_player_match_index(charted_matches),
        on=["match_id", "player"],
        how="inner",
        validate="many_to_one",
    )
    return indexed.loc[:, MCP_RALLY_COLUMNS]


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_wta_player_portrait(player_name: str) -> dict[str, str] | None:
    """Find the current official WTA portrait for an exact player-name match.

    The source datasets' player IDs are not WTA profile IDs, so this deliberately
    resolves the name through WTA first. Results are cached for one day to avoid
    a network request for each Streamlit rerun. No image is saved locally.
    """
    try:
        player_response = requests.get(
            f"{WTA_API_BASE}/tennis/players",
            params={"name": player_name},
            timeout=10,
        )
        player_response.raise_for_status()
        candidates = player_response.json().get("content", [])
    except (requests.RequestException, ValueError, AttributeError):
        return None

    expected_name = normalized_player_name(player_name)
    exact_matches = [
        candidate
        for candidate in candidates
        if normalized_player_name(str(candidate.get("fullName", ""))) == expected_name
    ]
    if len(exact_matches) != 1:
        return None

    wta_player = exact_matches[0]
    player_id = wta_player.get("id")
    if player_id is None:
        return None

    try:
        detail_response = requests.get(
            f"{WTA_API_BASE}/tennis/players/{player_id}/detailed",
            timeout=10,
        )
        detail_response.raise_for_status()
        wta_bio = detail_response.json().get("bio", {})
    except (requests.RequestException, ValueError, AttributeError):
        wta_bio = {}

    try:
        photo_response = requests.get(
            f"{WTA_API_BASE}/content/wta/PHOTO/en",
            params={
                "limit": WTA_PHOTO_QUERY_LIMIT,
                "tagNames": "player-headshot",
                "referenceExpression": f"TENNIS_PLAYER:{player_id}",
            },
            timeout=10,
        )
        photo_response.raise_for_status()
        photos = photo_response.json().get("content", [])
    except (requests.RequestException, ValueError, AttributeError):
        photos = []

    def has_tag(photo: dict, label: str) -> bool:
        return any(tag.get("label") == label for tag in photo.get("tags", []))

    # Prefer a close crop, then use a full-body headshot when that is all WTA has.
    portrait = next(
        (photo for photo in photos if has_tag(photo, "head-cropped-photo")),
        next((photo for photo in photos if has_tag(photo, "full-body-headshot")), None),
    )
    image_url = portrait.get("onDemandUrl") if portrait else None

    profile_slug = re.sub(
        r"[^a-z0-9]+",
        "-",
        unicodedata.normalize("NFKD", str(wta_player.get("fullName", player_name)))
        .encode("ascii", "ignore")
        .decode()
        .lower(),
    ).strip("-")
    result = {
        "player_id": str(player_id),
        "profile_url": f"https://www.wtatennis.com/players/{player_id}/{profile_slug}",
        "country_code": str(wta_player.get("countryCode", "")).upper(),
        "date_of_birth": str(wta_player.get("dateOfBirth", "")),
        "playing_hand": str(wta_bio.get("playhand", "")),
    }
    if image_url:
        result["image_url"] = resized_wta_photo_url(image_url)
    return result


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
    log["player_age"] = pd.to_numeric(
        matches["winner_age"].where(won, matches["loser_age"]), errors="coerce"
    )
    log["player_hand"] = matches["winner_hand"].where(won, matches["loser_hand"])
    log["opponent_hand"] = matches["loser_hand"].where(won, matches["winner_hand"])
    log["player_hand"] = log["player_hand"].map(normalized_hand)
    log["opponent_hand"] = log["opponent_hand"].map(normalized_hand)
    known_hands = log["player_hand"].notna() & log["opponent_hand"].notna()
    log["hand_matchup"] = pd.Series("Hand unavailable", index=log.index)
    log.loc[known_hands & log["player_hand"].eq(log["opponent_hand"]), "hand_matchup"] = (
        "Same-handed opponent"
    )
    log.loc[known_hands & log["player_hand"].ne(log["opponent_hand"]), "hand_matchup"] = (
        "Opposite-handed opponent"
    )
    # The source files do not identify confirmed injuries. A RET loss is the
    # closest player-specific observable, so it is retained as an explicit proxy.
    log["player_retired"] = (
        log["result"].eq("Loss")
        & matches["score_normalized"].str.contains(r"\bRET\b", regex=True, na=False)
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


def draw_match_charting_profile(log: pd.DataFrame, player: str) -> None:
    """Render matched Tennis Abstract shot-charting summaries for one player."""
    st.markdown("#### Tennis Abstract charted shot profile")
    winners = matched_shot_winner_stats(log, player)
    directions = matched_mcp_chart_rows(
        log, player, load_mcp_wta_shot_direction_stats()
    )
    shot_types = matched_mcp_chart_rows(log, player, load_mcp_wta_shot_type_stats())
    rallies = matched_mcp_chart_rows(log, player, load_mcp_wta_rally_stats())
    coverage_counts = {
        "Winners": winners["match_id"].nunique(),
        "Directions": directions["match_id"].nunique(),
        "Shot outcomes": shot_types["match_id"].nunique(),
        "Rallies": rallies["match_id"].nunique(),
    }
    if not any(coverage_counts.values()):
        st.info(
            "No uniquely matched Tennis Abstract charted matches are available for this player "
            "under the current filters. This means the project has no matching sample, not that "
            "the player recorded no shots."
        )
        return

    coverage = st.columns(4)
    for metric, column in zip(coverage_counts, coverage, strict=True):
        column.metric(f"{metric} matches", f"{coverage_counts[metric]:,}")

    if not winners.empty:
        winners["groundstroke_winners"] = winners["winners_fh"] + winners["winners_bh"]
        winner_averages = pd.DataFrame(
            {
                "Shot": ["Forehand", "Backhand", "Forehand + backhand"],
                "Average winners per match": [
                    winners["winners_fh"].mean(),
                    winners["winners_bh"].mean(),
                    winners["groundstroke_winners"].mean(),
                ],
            }
        )
        st.plotly_chart(
            px.bar(
                winner_averages,
                x="Shot",
                y="Average winners per match",
                text="Average winners per match",
                color="Shot",
                title="Average forehand and backhand winners per matched match",
                labels={"Average winners per match": "Winners / match"},
            ).update_traces(texttemplate="%{text:.1f}", textposition="outside"),
            width="stretch",
        )

    left, right = st.columns(2)
    with left:
        if directions.empty:
            st.info("No matched groundstroke-direction rows are available.")
        else:
            direction_labels = {
                "crosscourt": "Crosscourt",
                "down_middle": "Down middle",
                "down_the_line": "Down the line",
                "inside_out": "Inside-out",
                "inside_in": "Inside-in",
            }
            direction_profile = (
                directions.assign(
                    Stroke=directions["row"].map({"F": "Forehand", "B": "Backhand"})
                )
                .melt(
                    id_vars="Stroke",
                    value_vars=list(direction_labels),
                    var_name="Direction code",
                    value_name="Shots",
                )
                .assign(Direction=lambda frame: frame["Direction code"].map(direction_labels))
                .groupby(["Stroke", "Direction"], as_index=False)["Shots"]
                .sum()
            )
            direction_profile["Direction share (%)"] = 100 * direction_profile["Shots"] / (
                direction_profile.groupby("Stroke")["Shots"].transform("sum").replace(0, pd.NA)
            )
            st.plotly_chart(
                px.bar(
                    direction_profile,
                    x="Stroke",
                    y="Direction share (%)",
                    color="Direction",
                    barmode="stack",
                    hover_data={"Shots": ":,", "Direction share (%)": ":.1f"},
                    title="Groundstroke direction mix",
                ).update_yaxes(range=[0, 100], ticksuffix="%"),
                width="stretch",
            )
    with right:
        if shot_types.empty:
            st.info("No matched shot-outcome rows are available.")
        else:
            type_labels = {
                "Fgs": "Forehand groundstroke",
                "Bgs": "Backhand groundstroke",
                "Net": "Net shot",
            }
            outcomes = (
                shot_types.assign(Shot=shot_types["row"].map(type_labels))
                .groupby("Shot", as_index=False)[["shots", "winners", "unforced"]]
                .sum()
            )
            outcomes["Other"] = (
                outcomes["shots"] - outcomes["winners"] - outcomes["unforced"]
            ).clip(lower=0)
            outcome_profile = outcomes.melt(
                id_vars="Shot",
                value_vars=["winners", "unforced", "Other"],
                var_name="Outcome",
                value_name="Shots",
            ).replace({"winners": "Winner", "unforced": "Unforced error"})
            st.plotly_chart(
                px.bar(
                    outcome_profile,
                    x="Shot",
                    y="Shots",
                    color="Outcome",
                    barmode="stack",
                    title="Charted shot outcomes",
                    labels={"Shots": "Charted shots"},
                ),
                width="stretch",
            )

    if rallies.empty:
        st.info("No matched rally-length rows are available.")
    else:
        rally_order = ["1-3", "4-6", "7-9", "10"]
        rally_profile = (
            rallies.groupby("rally_band", as_index=False)[["points", "points_won"]]
            .sum()
            .assign(
                **{
                    "Point win rate (%)": lambda frame: 100
                    * frame["points_won"]
                    / frame["points"].replace(0, pd.NA)
                }
            )
        )
        st.plotly_chart(
            px.bar(
                rally_profile,
                x="rally_band",
                y="Point win rate (%)",
                text="points",
                category_orders={"rally_band": rally_order},
                title="Point win rate by rally length (shots)",
                labels={"rally_band": "Rally length", "points": "Charted points"},
            ).update_yaxes(range=[0, 100], ticksuffix="%"),
            width="stretch",
        )

    st.caption(
        "These charts use crowdsourced Tennis Abstract Match Charting Project records. "
        "Each metric retains only a unique player, opponent, tournament, and year match to the "
        "current combined log. Direction labels are charted categories—not ball-tracking or "
        "landing coordinates. Source: https://github.com/JeffSackmann/tennis_MatchChartingProject "
        "(CC BY-NC-SA 4.0; noncommercial use with attribution)."
    )


def draw_player_study(frame: pd.DataFrame, player: str) -> None:
    log = player_match_log(frame, player)
    if log.empty:
        st.info("No matches for this player match the current filters.")
        return

    portrait = get_wta_player_portrait(player)
    official_age = age_from_date_of_birth(
        portrait.get("date_of_birth") if portrait else None
    )
    recorded_ages = log.dropna(subset=["match_date", "player_age"]).sort_values("match_date")
    latest_recorded_age = (
        recorded_ages.iloc[-1]["player_age"] if not recorded_ages.empty else pd.NA
    )
    age_value = (
        f"{official_age} years"
        if official_age is not None
        else (
            f"{latest_recorded_age:.1f} years"
            if pd.notna(latest_recorded_age)
            else "N/A"
        )
    )
    age_help = (
        "Current age from the official WTA date of birth."
        if official_age is not None
        else "Age from the player's most recent match record in the selected data."
    )
    country_code = portrait.get("country_code", "") if portrait else ""
    flag_url = wta_flag_url(country_code) if country_code else None
    official_hand = normalized_hand(portrait.get("playing_hand", "") if portrait else "")
    detected_hands = log["player_hand"].dropna()
    source_hand = detected_hands.mode().iloc[0] if not detected_hands.empty else None
    player_hand = official_hand or source_hand
    player_retirements = int(log["player_retired"].sum())
    retirement_rate = pct(player_retirements, len(log))
    matchup_order = ["Same-handed opponent", "Opposite-handed opponent"]
    hand_matchups = log[log["hand_matchup"].isin(matchup_order)]
    hand_summary = (
        hand_matchups.groupby("hand_matchup", as_index=False)
        .agg(
            matches=("result", "size"),
            wins=("result", lambda results: (results == "Win").sum()),
        )
        .set_index("hand_matchup")
        .reindex(matchup_order, fill_value=0)
        .rename_axis("hand_matchup")
        .reset_index()
    )
    hand_summary["losses"] = hand_summary["matches"] - hand_summary["wins"]
    hand_summary["win_rate_pct"] = (
        100 * hand_summary["wins"] / hand_summary["matches"].replace(0, pd.NA)
    )

    def hand_matchup_metric(matchup: str) -> tuple[str, str]:
        row = hand_summary.loc[hand_summary["hand_matchup"].eq(matchup)].iloc[0]
        if not row["matches"]:
            return "N/A", "No matches with both hands recorded"
        return f"{row['win_rate_pct']:.1f}%", f"{int(row['wins'])}-{int(row['losses'])} ({int(row['matches'])} matches)"

    image_column, flag_column, title_column = st.columns([1, 0.95, 3.05])
    with image_column:
        if portrait and portrait.get("image_url"):
            st.image(
                portrait["image_url"],
                caption="Official WTA portrait",
                width="stretch",
            )
        else:
            st.info("Official WTA portrait unavailable for this player.")
    with flag_column:
        if flag_url:
            st.image(flag_url, width="stretch")
            st.caption(f"Represents {country_code}")
    with title_column:
        st.subheader(f"Player study: {player}")
        if portrait:
            st.link_button("Open official WTA profile ↗", portrait["profile_url"])
        st.caption(
            "Portrait, represented-country flag, date of birth, and playing hand are fetched from "
            "WTA’s official service and cached for one day; no image files are stored locally."
        )

    wins = int(log["result"].eq("Win").sum())
    completed = log["match_status"].eq("Completed")
    metrics = st.columns(6)
    metrics[0].metric("Age", age_value, help=age_help)
    metrics[1].metric("Combined matches", f"{len(log):,}")
    metrics[2].metric("Record", f"{wins}-{len(log) - wins}")
    metrics[3].metric("Win rate", f"{wins / len(log) * 100:.1f}%")
    metrics[4].metric("Completed matches", f"{completed.mean() * 100:.1f}%")
    metrics[5].metric(
        "Injury / retirement rate†",
        f"{retirement_rate:.1f}%",
        delta=f"{player_retirements} player retirement(s)",
        delta_color="off",
        help=(
            "The source files do not record confirmed injuries. This is the share of selected "
            "matches the player lost with RET in the score, which is an injury/illness proxy."
        ),
    )
    st.caption(
        "† The data does not identify confirmed injuries. A player retirement (RET) is used as "
        "an injury/illness proxy; opponent retirements are excluded."
    )

    st.markdown("#### Playing hand and opponent matchup")
    hand_metrics = st.columns(4)
    hand_metrics[0].metric(
        "Playing hand",
        hand_label(player_hand),
        help=(
            "Official WTA profile value when available; otherwise the most frequent hand value "
            "in the selected match records."
        ),
    )
    same_value, same_detail = hand_matchup_metric("Same-handed opponent")
    opposite_value, opposite_detail = hand_matchup_metric("Opposite-handed opponent")
    hand_metrics[1].metric(
        "Win rate vs same-handed",
        same_value,
        delta=same_detail,
        delta_color="off",
    )
    hand_metrics[2].metric(
        "Win rate vs opposite-handed",
        opposite_value,
        delta=opposite_detail,
        delta_color="off",
    )
    hand_metrics[3].metric(
        "Hand data coverage",
        f"{pct(len(hand_matchups), len(log)):.1f}%",
        delta=f"{len(hand_matchups):,} of {len(log):,} matches",
        delta_color="off",
        help="A comparison needs a recorded hand for both the player and opponent.",
    )
    st.plotly_chart(
        px.bar(
            hand_summary.dropna(subset=["win_rate_pct"]),
            x="hand_matchup",
            y="win_rate_pct",
            text="matches",
            color="hand_matchup",
            category_orders={"hand_matchup": matchup_order},
            labels={
                "hand_matchup": "Opponent hand matchup",
                "win_rate_pct": "Win rate (%)",
                "matches": "Matches",
            },
            title="Win rate by opponent hitting hand (label = matches)",
        ).update_yaxes(ticksuffix="%", range=[0, 100]),
        width="stretch",
    )

    draw_match_charting_profile(log, player)

    st.markdown("#### Per-match averages")
    average_minutes = log["minutes"].mean()
    average_aces = log["aces"].mean()
    average_double_faults = log["double_faults"].mean()
    average_first_serve_won = log["first_serve_won_pct"].mean()
    average_second_serve_won = log["second_serve_won_pct"].mean()
    average_temperature = log["weather_temperature_mean_f"].mean()
    average_metrics_top = st.columns(3)
    average_metrics_top[0].metric(
        "Average match duration",
        "N/A" if pd.isna(average_minutes) else f"{average_minutes:.1f} min",
    )
    average_metrics_top[1].metric(
        "Average aces", "N/A" if pd.isna(average_aces) else f"{average_aces:.1f}"
    )
    average_metrics_top[2].metric(
        "Average double faults",
        "N/A" if pd.isna(average_double_faults) else f"{average_double_faults:.1f}",
    )
    average_metrics_bottom = st.columns(3)
    average_metrics_bottom[0].metric(
        "Average 1st serve won",
        "N/A" if pd.isna(average_first_serve_won) else f"{average_first_serve_won:.1f}%",
    )
    average_metrics_bottom[1].metric(
        "Average 2nd serve won",
        "N/A" if pd.isna(average_second_serve_won) else f"{average_second_serve_won:.1f}%",
    )
    average_metrics_bottom[2].metric(
        "Average tournament-week temperature",
        "N/A" if pd.isna(average_temperature) else f"{average_temperature:.1f} °F",
    )

    st.markdown("#### Detailed player-relative match log")
    st.caption(
        "One de-duplicated row per match. Weather values are tournament-week aggregates when a venue or city mapping is available."
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
        "player_age",
        "player_hand",
        "opponent_hand",
        "hand_matchup",
        "minutes",
        "player_retired",
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
                "player_age": "Player age",
                "player_hand": "Player hand",
                "opponent_hand": "Opponent hand",
                "hand_matchup": "Opponent hand matchup",
                "player_retired": "Player retired (injury proxy)",
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
    st.divider()

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
    match_averages_by_year = (
        log.groupby("source_year", as_index=False)
        .agg(
            **{
                "Average duration (min)": ("minutes", "mean"),
                "Average aces": ("aces", "mean"),
                "Average double faults": ("double_faults", "mean"),
                "Average 1st serve won (%)": ("first_serve_won_pct", "mean"),
                "Average 2nd serve won (%)": ("second_serve_won_pct", "mean"),
                "Average temperature (°F)": ("weather_temperature_mean_f", "mean"),
            }
        )
    )
    serve_trend = match_averages_by_year.melt(
        id_vars="source_year",
        value_vars=["Average 1st serve won (%)", "Average 2nd serve won (%)"],
        var_name="Measure",
        value_name="Rate (%)",
    )
    workload_trend = match_averages_by_year.melt(
        id_vars="source_year",
        value_vars=["Average duration (min)", "Average aces", "Average double faults"],
        var_name="Measure",
        value_name="Average per match",
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
                serve_trend,
                x="source_year",
                y="Rate (%)",
                color="Measure",
                markers=True,
                labels={"source_year": "Year"},
                title="Average serve-point win rates by season",
            ).update_yaxes(ticksuffix="%", range=[0, 100]),
            width="stretch",
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.line(
                workload_trend,
                x="source_year",
                y="Average per match",
                facet_col="Measure",
                markers=True,
                labels={"source_year": "Year"},
                title="Average duration, aces, and double faults by season",
            ).update_yaxes(matches=None),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.line(
                match_averages_by_year,
                x="source_year",
                y="Average temperature (°F)",
                markers=True,
                labels={"source_year": "Year"},
                title="Average tournament-week temperature by season",
            ).update_yaxes(ticksuffix=" °F"),
            width="stretch",
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
    selected_player = player if player != "Select a player…" else None
    if selected_player:
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
