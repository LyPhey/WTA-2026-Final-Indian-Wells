from step_02_load_and_compatre_providers import load_provider

# This pytho file should answer these questions separately for Sackmann and TennisMyLife:

# 1. Are key fields missing?
# tourney_date, winner_name, loser_name, score, surface, rankings, serve statistics.

sackmann = load_provider('sackmann', 'wta_matches_*.csv')
tennismylife = load_provider('tennismylife', '*_wta.csv')

print(sackmann)
print(tennismylife)

# to visualize the sackmann columns
print("\nSackmann columns:")
print(sackmann.columns.tolist())

# to visualize the tennisislife columns
print("\nTennisIsLife columns:")
print(tennismylife.columns.tolist())

# to inspect just the sackmann 2026 matches
sackmann_2026 = sackmann[sackmann['source_year'] == 2026]
print(sackmann_2026.head().to_string(index = False))

# to find matches involving a particular player
player = "Coco Gauff"

coco_match = sackmann[
    (sackmann['winner_name'] == player) | (sackmann['loser_name'] == player)]

print(coco_match[
    ['tourney_name', 'tourney_date', 'winner_name', 'loser_name', 'score', 'surface']
].head(10).to_string(index = False))