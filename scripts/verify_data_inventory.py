from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs" / "DATA_INVENTORY.csv"
FETCH_MANIFEST = ROOT / "docs" / "DATA_FETCH.csv"
DATA_DIR = ROOT / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local data against the reviewed inventory.")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Treat inventory entries missing from data/ as failures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    missing: list[str] = []
    checked = 0

    with INVENTORY.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with FETCH_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        modes = {row["file"]: row["mode"] for row in csv.DictReader(handle)}

    inventory_names = {row["file"] for row in rows}
    non_data_files = {"README.md"}
    if DATA_DIR.exists():
        unexpected = sorted(
            path.name
            for path in DATA_DIR.iterdir()
            if path.is_file() and path.name not in inventory_names and path.name not in non_data_files
        )
        failures.extend(f"unexpected file: {name}" for name in unexpected)

    for row in rows:
        path = DATA_DIR / row["file"]
        if not path.exists():
            missing.append(row["file"])
            continue

        checked += 1
        mode = modes[row["file"]]
        actual_size = path.stat().st_size
        if mode not in {"download", "repository"}:
            if actual_size == 0:
                failures.append(f"empty generated or user-supplied file: {row['file']}")
            continue
        expected_size = int(row["size_bytes"])
        if actual_size != expected_size:
            failures.append(f"size mismatch: {row['file']} ({actual_size} != {expected_size})")
            continue

        actual_hash = sha256(path)
        if actual_hash != row["sha256"]:
            failures.append(f"hash mismatch: {row['file']}")

    if args.require_all:
        required_missing = [name for name in missing if modes[name] not in {"manual", "excluded"}]
        failures.extend(f"missing file: {name}" for name in required_missing)

    print(f"Inventory entries: {len(rows)}")
    print(f"Checked local files: {checked}")
    print(f"Missing local files: {len(missing)}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1

    print("Inventory verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
