from pathlib import Path
import json
import pandas as pd
import re

from environmental_data_pipeline import (
    AUTO_GEOCODED_LOCATIONS_PATH,
    LIVE_TENNIS_LOCATIONS_PATH,
    TOURNAMENT_WEEK_WEATHER_PATH,
    VENUE_REFERENCE_PATH,
    VENUE_TEMPLATE_PATH,
)

def year_from_filename(path: Path) -> int:
    match = re.search(r"(?:19|20)\d{2}", path.stem)
    if not match:
        raise ValueError(f"No year found in filename: {path.name}")
    return int(match.group())



RAW = Path('data/raw')

files = (
    list((RAW / "sackmann").glob("wta_matches_*.csv")) +
    list((RAW / "tennismylife").glob("*_wta.csv"))
)

inventory = []

for path in sorted(files):
    source = path.parent.name
    year = year_from_filename(path)
    header = pd.read_csv(path, nrows = 0).columns.tolist()

    inventory.append({
        "Provider": source,
        "year": year,
        "file": path.name,
        "columns": len(header),
        "rows": sum(1 for _ in open(path, encoding="utf-8-sig")) - 1,
        "has_indoor": "indoor" in header,
        "has_serve_stats": {"w_svpt", "l_svpt"}.issubset(header),
        "has_ranking": {"winner_rank", "loser_rank"}.issubset(header),
    })

inventory_df = (
    pd.DataFrame(inventory)
    .sort_values(["Provider", "year"])
    .reset_index(drop = True) 
)

print(inventory_df.to_string(index = False))


environmental_assets = []
for label, path in {
    "venue reference": VENUE_REFERENCE_PATH,
    "venue review template": VENUE_TEMPLATE_PATH,
    "published tournament-city locations": LIVE_TENNIS_LOCATIONS_PATH,
    "automatic geocoding candidates": AUTO_GEOCODED_LOCATIONS_PATH,
    "tournament-week weather": TOURNAMENT_WEEK_WEATHER_PATH,
}.items():
    if path.exists():
        if path.suffix == ".geojson":
            payload = json.loads(path.read_text(encoding="utf-8"))
            features = payload.get("features", [])
            property_names = sorted(
                {name for feature in features for name in feature.get("properties", {})}
            )
            header = ["geometry", *property_names]
            row_count = len(features)
        else:
            header = pd.read_csv(path, nrows=0).columns.tolist()
            row_count = sum(1 for _ in path.open(encoding="utf-8-sig")) - 1
        environmental_assets.append({
            "asset": label,
            "file": str(path),
            "available": True,
            "columns": len(header),
            "rows": row_count,
        })
    else:
        environmental_assets.append({
            "asset": label,
            "file": str(path),
            "available": False,
            "columns": 0,
            "rows": 0,
        })

print("\nEnvironmental-pipeline assets:")
print(pd.DataFrame(environmental_assets).to_string(index=False))

if TOURNAMENT_WEEK_WEATHER_PATH.exists():
    weather = pd.read_csv(TOURNAMENT_WEEK_WEATHER_PATH, low_memory=False)
    if "weather_status" in weather:
        print("\nTournament-week weather status:")
        print(weather["weather_status"].value_counts(dropna=False).to_string())


