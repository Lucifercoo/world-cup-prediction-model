# Data Directory

Run `uv run python -m wc_model setup` for the complete public data and model
pipeline. Run `uv run python scripts/fetch_data.py --list` to inspect every
file and its acquisition mode, then
`uv run python scripts/verify_data_inventory.py` to verify local checksums.

Project-authored records committed here are licensed under CC BY 4.0. Each
third-party dataset retains its upstream license. See
[`DATA_LICENSES.md`](../DATA_LICENSES.md) and
[`docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md).

Public setup uses a documented FIFA-points squad-value proxy and an empty
optional key-player configuration. These inputs are labeled in their source
fields and are not presented as Transfermarkt data or formal-run inputs.

`strict_pre_match_predictions.csv` contains 79 project-generated predictions
selected strictly before kickoff, their base-model counterparts, and the
realtime audit fields used by the assisted system. It contains no results or
hit labels. `uv run python -m wc_model evaluate` joins it to the independent
result ledger and rebuilds both model-only and realtime-assisted metrics. Its
manifest records GPT-5.5 / `very_high` as the historical realtime research
environment, plus the archive hash and hashes of the 26 original cache runs.
