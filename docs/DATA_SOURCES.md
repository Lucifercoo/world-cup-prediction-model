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

Download every immutable and redistributable source currently supported:

```powershell
uv run python scripts/fetch_data.py
```

After manually supplying permitted prerequisites, generated datasets can be
built with `uv run python scripts/fetch_data.py --build`. Every downloaded file
is checked against the recorded size and SHA-256 before it replaces a local
file. Run `uv run python scripts/verify_data_inventory.py` for a full local
inventory check.

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
| [cashncarry/fifaworldranking](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) | Historical FIFA ranking export | CC0 on the dataset page | Review the local file's provenance before inclusion. |
| [Dato-Futbol/fifa-ranking](https://github.com/Dato-Futbol/fifa-ranking) | Historical FIFA points | No repository license | Exclude the copied data; keep source instructions only. |
| [FIFA World Ranking](https://inside.fifa.com/fifa-world-ranking/men) | 2025/2026 snapshots | [FIFA Terms of Service](https://legal.fifa.com/terms-of-service) do not grant general dataset redistribution | Exclude raw and derived snapshots. |
| [Open-Meteo](https://open-meteo.com/) | Historical weather and geocoding | [CC BY 4.0 data; API plan limits apply](https://open-meteo.com/en/terms) | Rebuild caches and attribute Open-Meteo. |
| [Wikipedia: 2026 FIFA World Cup squads](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads) | Player-club affiliations | CC BY-SA 4.0 | Include only with article attribution and share-alike notice. |
| [Transfermarkt](https://www.transfermarkt.us/world-cup/teilnehmer/pokalwettbewerb/FIWC) | Squad market values | [Terms prohibit automated scraping/copying](https://www.transfermarkt.us/intern/anb) | Exclude scraped values and the automated-fetch workflow from a public release. |

## Reproducibility strategy

1. Commit redistributable source data with source, license, retrieval date, and
   checksum.
2. Commit project-authored match inputs and generated state only after checking
   that notes summarize facts rather than copy article prose.
3. Replace restricted files with schemas and user-supplied input paths. A clean
   checkout must fail clearly when a required restricted input is missing.
4. Keep API caches out of Git. Builders must record endpoint, request date,
   parameters, and attribution in their generated manifest.
5. Never replace unavailable production data with mock values.

This inventory is a conservative engineering review, not legal advice.

## Restricted input schemas

The repository provides header-only schemas for required inputs that cannot be
redistributed:

- `schemas/fifa_rankings_annual_start.csv`
- `schemas/transfermarkt_world_cup_2026_values.csv`
- `schemas/world_cup_2026_key_player_signals.csv`

Place legally obtained files with those names in `data/`. Empty schemas are
not valid model inputs and are never used as fallback data.
