# Data Licensing

Code in this repository is licensed under the [MIT License](LICENSE). Data is
licensed separately according to its source.

## Project-authored data

The following files and project-generated derivatives are licensed under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/):

- `data/match_shape_context.csv`
- `data/realtime_team_context.csv`
- `data/world_cup_2026_key_player_match_status.csv`
- `data/world_cup_2026_knockout_decisions.csv`
- `data/world_cup_2026_result_sources.csv`
- `data/world_cup_2026_results.csv`
- `data/world_cup_2026_team_discipline_profiles.csv`
- `data/strict_pre_match_predictions.csv` and its provenance manifest
- generated tournament-adjustment and style-profile datasets identified as
  `include-project` in `docs/DATA_INVENTORY.csv`
- the public-mode squad-value proxy generated from pre-tournament FIFA points

Attribution: **World Cup Prediction Model data, Lucifercoo, 2026**. State when
changes have been made and retain links to the original reporting sources.

## Third-party data

- `international_results.csv`: CC0-1.0, from
  [martj42/international_results](https://github.com/martj42/international_results).
- `fifa_rankings_history_open.csv`: normalized locally from the pinned
  `cashncarry/fifaworldranking` file identified as CC0 on its dataset page.
- `world_cup_2026_squad_clubs.csv` and
  `world_cup_2026_team_club_cohesion.csv`: derived from
  [Wikipedia's 2026 FIFA World Cup squads article](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads)
  under CC BY-SA 4.0. Modified into normalized tabular and aggregate forms.
- Open-Meteo cache data: CC BY 4.0; see
  [Open-Meteo terms](https://open-meteo.com/en/terms).

Locally generated FIFA snapshots remain subject to FIFA's terms and are not
redistributed by this project. Users replacing public proxy inputs remain
responsible for the terms governing their own data.
