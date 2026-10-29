from pathlib import Path
import re
import pandas as pd

from environmental_data_pipeline import (
    environmental_coverage,
    enrich_matches_with_environment,
    load_tournament_week_weather,
)

RAW = Path('data/raw')

def year_from_filename(path: Path) -> int:
    match = re.search(r"(?:19|20)\d{2}", path.stem)
    if not match:
        raise ValueError(f"No year found in filename: {path.name}")
    return int(match.group())

def load_provider(
    provider: str, pattern: str, include_environment: bool = False
) -> pd.DataFrame:
    """Load one provider, optionally joined to cached tournament-week weather."""
    paths = sorted((RAW / provider).glob(pattern))

    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory = False)
        frame['provider'] = provider
        frame['source_file'] = path.name
        frame['source_year'] = year_from_filename(path)
        frames.append(frame)

    matches = pd.concat(frames, ignore_index=True)
    return enrich_matches_with_environment(matches) if include_environment else matches


def load_provider_with_environment(provider: str, pattern: str) -> pd.DataFrame:
    """Convenience wrapper for source-level environmental enrichment."""
    return load_provider(provider, pattern, include_environment=True)

if __name__ == "__main__":
    sackmann = load_provider('sackmann', 'wta_matches_*.csv')
    tennismylife = load_provider('tennismylife', '*_wta.csv')

    print('Sackmann:', sackmann.shape)
    print('TennisMyLife:', tennismylife.shape)

    only_sackmann = sorted(set(sackmann.columns) - set(tennismylife.columns))
    only_tml = sorted(set(tennismylife.columns) - set(sackmann.columns))

    print('Only in Sackmann:', only_sackmann)
    print("Only in TennisMyLife:", only_tml)

    weather = load_tournament_week_weather()
    print(f"Cached tournament-week weather rows: {len(weather):,}")
    enriched = pd.concat(
        [
            enrich_matches_with_environment(sackmann),
            enrich_matches_with_environment(tennismylife),
        ],
        ignore_index=True,
    )
    print("Environmental coverage:")
    print(environmental_coverage(enriched).to_string(index=False))



