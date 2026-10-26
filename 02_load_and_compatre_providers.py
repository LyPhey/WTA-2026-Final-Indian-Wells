from pathlib import Path
import re
import pandas as pd

RAW = Path('data/raw')

def year_from_filename(path: Path) -> int:
    match = re.search(r"(?:19|20)\d{2}", path.stem)
    if not match:
        raise ValueError(f"No year found in filename: {path.name}")
    return int(match.group())

def load_provider(provider: str, pattern: str) -> pd.DataFrame:
    paths = sorted((RAW / provider).glob(pattern))

    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory = False)
        frame['provider'] = provider
        frame['source_file'] = path.name
        frame['source_year'] = year_from_filename(path)
        frames.append(frame)

    return pd.concat(frames, ignore_index = True)

sackmann = load_provider('sackmann', 'wta_matches_*.csv')
tennismylife = load_provider('tennismylife', '*_wta.csv')

print('Sackmann:', sackmann.shape)
print('TennisMyLife:', tennismylife.shape)

only_sackmann = sorted(set(sackmann.columns) - set(tennismylife.columns))
only_tml = sorted(set(tennismylife.columns) - set(sackmann.columns))

print('Only in Sackmann:', only_sackmann)
print("Only in TennisMyLife:", only_tml)



