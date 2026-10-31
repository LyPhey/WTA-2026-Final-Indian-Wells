# WTA match-prediction dataset starter

Updated on 2026-10-25 for a **noncommercial research** project predicting WTA match winners, exact best-of-three set scores (2-0, 2-1, 1-2, 0-2), and upset likelihood for the 2026 WTA Finals at Indian Wells.

## What is included

- `download_data.py`: reproducible downloading of **2016–2026 WTA CSVs** from TennisMyLife and the **archived Jeff Sackmann** GitHub mirror; by default includes the WTA player file and TennisMyLife current-tournaments file. The two sources remain **separate**.
- `audit_data.py`: preliminary file-level missingness, duplication and player coverage reports. Checks how much serve/ranking information is populated; does **not** certify factual correctness.
- `environmental_data_pipeline.py`: creates a reviewable venue map, fetches Open-Meteo historical weather, and joins safe tournament-week aggregates to the source-level matches.
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

### Environmental enrichment

The match feeds do not provide court coordinates, match dates, or match start
times. The environmental pipeline first uses a reviewed venue map, then an
unambiguous published tournament-city map, and aggregates Open-Meteo ERA5
reanalysis data across the seven days starting at `tourney_date`. Temperatures
are stored in Fahrenheit. It is **tournament-week weather**, not observed
match-day weather.

```bash
# Generate the review template for all tournaments in the raw sources.
python environmental_data_pipeline.py build-venue-template

# Sync published city-level locations, cache weather, and rebuild matches.
python environmental_data_pipeline.py run
```

`data/reference/venue_reference.csv` includes an approved Indian Wells Tennis
Garden mapping as an example. `data/reference/venue_reference_template.csv` is
generated for review and must not be treated as validated location data. The
pipeline caches city-level location candidates that need review separately and
leaves events with no reliable location as `unavailable_no_mapped_location`—it
never assigns weather to an inferred host city. The weather source is the
[Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), using ERA5 reanalysis. The published city map is from [Live Tennis API](https://livetennisapi.com/tools/tournament-map). Keep both sources' CC BY 4.0 attributions in downstream outputs.

### Canonical combined match file

After downloading the raw sources and building the cached weather enrichment,
create the ready-to-use, de-duplicated match table with:

```bash
python build_combined_dataset.py
```

This writes `data/processed/wta_matches_combined_deduplicated.csv`. It retains
one record for every normalized tournament-week, tournament, round, winner,
and loser identity. It preserves the original raw inputs and records the
selected source in `retained_provider`, the contributing providers in
`source_providers`, their count in `source_record_count`, and the applied
deduplication rule. It includes environmental fields, but does not create any
model features or train/test splits.

### Leakage-safe pre-match features and baseline

Create the historical training table with:

```bash
python step_05_build_pre_match_features.py
```

This writes `data/processed/wta_pre_match_features.csv`. Each row is a
neutral Player A versus Player B record: Player A is selected alphabetically,
not because she won. Historical form, serve, and head-to-head features use
only completed matches from strictly earlier tournament weeks. Current-match
score, duration, serve statistics, winner/loser fields, source-coverage
fields, and retrospective weather are not included as model inputs.

Because the match sources provide tournament-week dates rather than reliable
match dates, the feature builder deliberately excludes all same-week results.
This is conservative and prevents a later match in a week from accidentally
learning from an earlier result whose timing is unknown.

To fit an interpretable baseline after the feature table is built:

```bash
python step_06_train_baseline_model.py
```

The default baseline fits logistic regression through 2024 and evaluates once
on 2025. It stores the fitted pipeline in `models/` and held-out accuracy,
Brier score, log loss, and ROC AUC in `reports/`. It deliberately excludes
2026 so that season remains available for a future pre-WTA-Finals retraining
cutoff.

On some networks TennisMyLife may block scripted CSV access; if you encounter an error, open https://stats.tennismylife.org/tennis-match-database and download WTA years manually into `data/raw/tennismylife/`. TennisData.App CSVs may similarly need manual browser downloads, saved separately to `data/raw/tennisdata_app/` (which this downloader intentionally does not scrape).

### Keeping data current

Re-run `python download_data.py --provider tennismylife --overwrite` for new 2026 source files. Log the date of the update. DO NOT overwrite your evaluation snapshots used for earlier match predictions: pre-match forecasting requires the information available *at that time*.

## Dataset boundaries / modeling cautions

1. **Cutoff:** Predicting a 2026-11-08 match means using historical information available strictly beforehand. The 2026 files can change after this date, so snapshot them before predictions and freeze the raw files.
2. **Dates:** `tourney_date` is usually the *tournament week's date*, not the actual day of each match. It cannot reliably produce rest-day or match-day weather features; find a separately permitted match-date source before using them.
3. **Leakage:** Data comes as `winner_*` and `loser_*`; create player-A/player-B rows *without defining A as the winner*. Any service stats from the predicted match are future information; only rolling averages of prior matches can be used.
4. **Retirements:** Separate walkovers, retirements, defaults and incomplete matches from ordinary best-of-three set-score training.
5. **Duplication:** TennisMyLife and Sackmann have overlapping records; don't simply concatenate. Match on normalized player identity, tournament, year, round, and score then inspect discrepancies.
6. **Elo/Glicko:** Current published ratings are useful on prediction day, but using today's rating as a historical feature for 2022 training data is lookahead bias. Recompute ratings chronologically.
7. **Odds:** If market odds are used to define an upset, make sure they were available *before* the match, and check provider terms.
8. **Licensing:** Sackmann's archive is CC BY-NC-SA 4.0; do not use it in a commercial product. TennisMyLife's website marks MIT, while its GitHub repository says noncommercial unless permission is obtained. Don't presume commercial rights or redistribute raw archives.
9. **Quality:** CSV presence, header and structural checks do not prove complete match coverage. Cross-validate records with official WTA results or an independently licensed provider.
10. **Weather leakage:** Historical reanalysis describes conditions after the fact. Do not use it as a pre-match prediction input. For operational forecasting, use only a weather forecast archived before the match and retain its issue time.

## Modeling sequence

1. Audit and resolve match identities; keep clean raw snapshots.
2. Build chronological neutral-player labels and pre-match features: overall/hard Elo, rolling performance in last 10/20 matches, prior service/return %, matchup history, tournament level.
3. Baseline: Elo and logistic regression; assess log loss, Brier score and reliability diagrams.
4. Compare CatBoost/XGBoost with walk-forward validation.
5. For set scores, start with a separately calibrated set-win probability / best-of-three probability model; check that four score probabilities sum to one and match the winner probability.
6. Upset risk = calibrated probability of the objectively defined pre-match underdog winning (by Elo, rankings, or legitimately available odds).

## Why the package contains downloader scripts, not CSVs

The source sites publish datasets, but they could not be transferred directly into the assistant's working environment. This kit is **not** a claim that the raw datasets have already been downloaded or audited. Run the scripts locally to retrieve the actual licensed source data and generate audit reports.
