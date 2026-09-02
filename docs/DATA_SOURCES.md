# Data Sources and Publication Policy

The model uses third-party datasets, API responses, project-authored match
records, and generated state. Source code licensing does not override the
terms attached to those inputs. The repository therefore treats every file in
`data/` as private by default and publishes it only after a file-level review.

The authoritative review is [`DATA_INVENTORY.csv`](DATA_INVENTORY.csv), while
[`DATA_FETCH.csv`](DATA_FETCH.csv) records the machine-readable acquisition
method. List the available methods with:

```powershell
uv run python scripts/fetch_data.py --list
```

Prepare all public-mode inputs and run the model:

```powershell
uv run python -m wc_model setup
```

Use `--data-only` to stop after data preparation. Immutable downloads and the
pinned Kaggle member are checked against recorded SHA-256 values. Run
`uv run python scripts/verify_data_inventory.py` for a full local inventory
check.

## Publication decisions

| Decision | Meaning |
| --- | --- |
| `include` | May be published with the stated attribution and license notice. |
| `include-project` | Project-authored facts, labels, or generated state; publish under the eventual project data license. |
| `rebuild` | Do not commit the local cache; document a reproducible fetch/build step. |
| `exclude` | Do not publish the local file because redistribution is prohibited or unsupported. |
| `review` | Keep private until provenance or copied text has been checked manually. |

## Upstream sources

| Source | Local use | License or terms | Current decision |
| --- | --- | --- | --- |
| [martj42/international_results](https://github.com/martj42/international_results) | Historical international results | CC0-1.0 | Include with provenance. |
| [cashncarry/fifaworldranking](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | Historical FIFA ranking export | CC0 on the dataset page | Download a pinned member and normalize locally. |
| [FIFA World Ranking](https://inside.fifa.com/fifa-world-ranking/men) | 2025/2026 snapshots | [FIFA Terms of Service](https://legal.fifa.com/terms-of-service) do not grant general dataset redistribution | Generate locally; do not commit snapshots. |
| [Open-Meteo](https://open-meteo.com/) | Historical weather and geocoding | [CC BY 4.0 data; API plan limits apply](https://open-meteo.com/en/terms) | Rebuild caches and attribute Open-Meteo. |
| [Wikipedia: 2026 FIFA World Cup squads](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads) | Player-club affiliations | CC BY-SA 4.0 | Include only with article attribution and share-alike notice. |
| [Transfermarkt](https://www.transfermarkt.us/world-cup/teilnehmer/pokalwettbewerb/FIWC) | Formal-run squad market values | [Terms prohibit automated scraping/copying](https://www.transfermarkt.us/intern/anb) | Exclude; public mode uses a clearly labeled FIFA-points proxy. |

## Reproducibility strategy

1. Commit redistributable source data with source, license, retrieval date, and
   checksum.
2. Commit project-authored match inputs and generated state only after checking
   that notes summarize facts rather than copy article prose.
3. Build a public model variant from pinned historical rankings, fixed
   pre-tournament snapshots, and an explicitly labeled squad-value proxy.
4. Publish the minimum project-authored pre-match prediction archive needed to
   recompute formal metrics; keep results and hit labels in independent files.
5. Keep API caches out of Git. Builders must record endpoint, request date,
   parameters, and attribution in their generated manifest.
6. Never replace unavailable production data with mock values.

This inventory is a conservative engineering review, not legal advice.

## Input schemas

The repository provides schemas for generated public inputs and permitted
user-supplied replacements:

- `schemas/fifa_rankings_annual_start.csv`
- `schemas/transfermarkt_world_cup_2026_values.csv`
- `schemas/world_cup_2026_key_player_signals.csv`

`world_cup_2026_key_player_signals.csv` may contain only its header; that
explicitly disables the optional key-player layer. FIFA ranking and squad-value
inputs must contain data rows. User-supplied replacements must retain the same
schema and comply with their upstream terms.
