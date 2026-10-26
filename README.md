# WTA match-prediction dataset starter

Updated on 2026-10-25 for a **noncommercial research** project predicting WTA match winners, exact best-of-three set scores (2-0, 2-1, 1-2, 0-2), and upset likelihood for the 2026 WTA Finals at Indian Wells.

## What is included

- `download_data.py`: reproducible downloading of **2016–2026 WTA CSVs** from TennisMyLife and the **archived Jeff Sackmann** GitHub mirror; by default includes the WTA player file and TennisMyLife current-tournaments file. The two sources remain **separate**.
- `audit_data.py`: preliminary file-level missingness, duplication and player coverage reports. Checks how much serve/ranking information is populated; does **not** certify factual correctness.
- `SOURCE_INVENTORY.csv`: source links, coverage and license caution notes.
- `requirements.txt`.

**Version 2 correction:** The downloader now recognizes the actual Sackmann
`wta_players.csv` (`first_name`, `last_name`) and historical ranking CSV schema,
so it won't falsely reject these as non-CSV downloads.

**If the ZIP link doesn't open:** Download individual files from the assistant's
message and put them together in a folder; they work without the ZIP. The
`--dry-run` command needs no external network access.

## Start here

From this folder, with Python 3.10+ and internet:

```bash
python -m pip install -r requirements.txt
python download_data.py --dry-run
python download_data.py --start-year 2016 --end-year 2026 --rankings
python audit_data.py
```

On some networks TennisMyLife may block scripted CSV access; if you encounter an error, open https://stats.tennismylife.org/tennis-match-database and download WTA years manually into `data/raw/tennismylife/`. TennisData.App CSVs may similarly need manual browser downloads, saved separately to `data/raw/tennisdata_app/` (which this downloader intentionally does not scrape).

### Keeping data current

Re-run `python download_data.py --provider tennismylife --overwrite` for new 2026 source files. Log the date of the update. DO NOT overwrite your evaluation snapshots used for earlier match predictions: pre-match forecasting requires the information available *at that time*.

## Dataset boundaries / modeling cautions

1. **Cutoff:** Predicting a 2026-11-08 match means using historical information available strictly beforehand. The 2026 files can change after this date, so snapshot them before predictions and freeze the raw files.
2. **Dates:** `tourney_date` is usually the *tournament week's date*, not the actual day of each match. It cannot reliably produce rest-day features; find a separately permitted match-date source before using them.
3. **Leakage:** Data comes as `winner_*` and `loser_*`; create player-A/player-B rows *without defining A as the winner*. Any service stats from the predicted match are future information; only rolling averages of prior matches can be used.
4. **Retirements:** Separate walkovers, retirements, defaults and incomplete matches from ordinary best-of-three set-score training.
5. **Duplication:** TennisMyLife and Sackmann have overlapping records; don't simply concatenate. Match on normalized player identity, tournament, year, round, and score then inspect discrepancies.
6. **Elo/Glicko:** Current published ratings are useful on prediction day, but using today's rating as a historical feature for 2022 training data is lookahead bias. Recompute ratings chronologically.
7. **Odds:** If market odds are used to define an upset, make sure they were available *before* the match, and check provider terms.
8. **Licensing:** Sackmann's archive is CC BY-NC-SA 4.0; do not use it in a commercial product. TennisMyLife's website marks MIT, while its GitHub repository says noncommercial unless permission is obtained. Don't presume commercial rights or redistribute raw archives.
9. **Quality:** CSV presence, header and structural checks do not prove complete match coverage. Cross-validate records with official WTA results or an independently licensed provider.

## Modeling sequence

1. Audit and resolve match identities; keep clean raw snapshots.
2. Build chronological neutral-player labels and pre-match features: overall/hard Elo, rolling performance in last 10/20 matches, prior service/return %, matchup history, tournament level.
3. Baseline: Elo and logistic regression; assess log loss, Brier score and reliability diagrams.
4. Compare CatBoost/XGBoost with walk-forward validation.
5. For set scores, start with a separately calibrated set-win probability / best-of-three probability model; check that four score probabilities sum to one and match the winner probability.
6. Upset risk = calibrated probability of the objectively defined pre-match underdog winning (by Elo, rankings, or legitimately available odds).

## Why the package contains downloader scripts, not CSVs

The source sites publish datasets, but they could not be transferred directly into the assistant's working environment. This kit is **not** a claim that the raw datasets have already been downloaded or audited. Run the scripts locally to retrieve the actual licensed source data and generate audit reports.
