# Data Directory

Run `uv run python scripts/fetch_data.py --list` to inspect every required
file and its acquisition mode. Run `uv run python scripts/fetch_data.py` to
download immutable redistributable sources, then
`uv run python scripts/verify_data_inventory.py` to verify local checksums.

Project-authored records committed here are licensed under CC BY 4.0. Each
third-party dataset retains its upstream license. See
[`DATA_LICENSES.md`](../DATA_LICENSES.md) and
[`docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md).

Restricted inputs are intentionally absent. Builders fail when those inputs
are missing; the project does not substitute mock data.
