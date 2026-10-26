from pathlib import Path
import pandas as pd
import re

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


