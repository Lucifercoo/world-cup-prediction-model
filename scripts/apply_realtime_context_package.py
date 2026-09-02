from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.prepare_realtime_context_package import (
    KEY_PLAYER_FIELDS,
    SHAPE_FIELDS,
    TEAM_FIELDS,
    validate_and_convert,
)

ROOT = Path(__file__).resolve().parents[1]

TABLES = {
    "realtime_team_context.csv": (TEAM_FIELDS, ("date_bjt", "time_bjt", "match", "team")),
    "match_shape_context.csv": (SHAPE_FIELDS, ("match",)),
    "world_cup_2026_key_player_match_status.csv": (
        KEY_PLAYER_FIELDS,
        ("date_bjt", "time_bjt", "match", "team", "key_player"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a validated realtime context package.")
    parser.add_argument("package_dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    return parser.parse_args()


def read_rows(path: Path, expected_fields: list[str], *, allow_missing: bool) -> list[dict[str, str]]:
    if not path.is_file():
        if allow_missing:
            return []
        raise ValueError(f"required package file is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise ValueError(f"unexpected fields in {path}: {reader.fieldnames}")
        return list(reader)


def row_key(row: dict[str, str], fields: tuple[str, ...]) -> tuple[str, ...]:
    key = tuple(row[field].strip() for field in fields)
    if any(not value for value in key):
        raise ValueError(f"empty key field {fields}: {row}")
    return key


def merge_rows(
    existing: list[dict[str, str]],
    incoming: list[dict[str, str]],
    key_fields: tuple[str, ...],
) -> tuple[list[dict[str, str]], int, int]:
    incoming_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for row in incoming:
        key = row_key(row, key_fields)
        if key in incoming_by_key:
            raise ValueError(f"duplicate package key {key_fields}={key}")
        incoming_by_key[key] = row

    merged: list[dict[str, str]] = []
    replaced = 0
    existing_keys: set[tuple[str, ...]] = set()
    for row in existing:
        key = row_key(row, key_fields)
        if key in existing_keys:
            raise ValueError(f"duplicate existing key {key_fields}={key}")
        existing_keys.add(key)
        replacement = incoming_by_key.get(key)
        if replacement is not None:
            merged.append(replacement)
            replaced += 1
        else:
            merged.append(row)
    merged.extend(row for key, row in incoming_by_key.items() if key not in existing_keys)
    return merged, len(incoming_by_key) - replaced, replaced


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def apply_package(package_dir: Path, data_dir: Path) -> list[tuple[str, int, int]]:
    source_json = package_dir / "context_package.json"
    if not source_json.is_file():
        raise ValueError(f"context_package.json is missing from {package_dir}")
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    validate_and_convert(payload, package_dir)

    prepared: list[tuple[Path, list[str], list[dict[str, str]]]] = []
    summary: list[tuple[str, int, int]] = []
    for name, (fields, key_fields) in TABLES.items():
        incoming = read_rows(package_dir / name, fields, allow_missing=False)
        target = data_dir / name
        existing = read_rows(target, fields, allow_missing=True)
        merged, inserted, replaced = merge_rows(existing, incoming, key_fields)
        prepared.append((target, fields, merged))
        summary.append((name, inserted, replaced))

    for target, fields, rows in prepared:
        write_rows(target, fields, rows)
    return summary


def main() -> int:
    args = parse_args()
    for name, inserted, replaced in apply_package(args.package_dir, args.data_dir):
        print(f"{name}: inserted={inserted}, replaced={replaced}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
