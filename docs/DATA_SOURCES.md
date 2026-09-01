# Data Sources

This project combines public match results, FIFA ranking snapshots, squad market
values, squad club affiliations, weather context, and manually reviewed match
context. Source-specific terms still apply to every copied or generated dataset.

| Dataset | Purpose | Upstream source |
| --- | --- | --- |
| `international_results.csv` | Rolling team profiles and historical backtests | Public international match results dataset |
| `fifa_rankings_*.csv/json` | FIFA ranking snapshots | FIFA and archived ranking datasets |
| `transfermarkt_world_cup_2026_values.csv` | Squad-value signal | Transfermarkt public pages |
| `world_cup_2026_squad_clubs.csv` | Club concentration signal | Public squad lists and Wikipedia |
| `open_meteo_*.json` | Weather and geocoding cache | Open-Meteo |
| `world_cup_2026_result_sources.csv` | Result provenance | Linked public reports |

Before publishing a release, verify whether each upstream source permits
redistribution. Files that cannot be redistributed should be replaced by fetch
instructions or excluded from Git while preserving their schema.

