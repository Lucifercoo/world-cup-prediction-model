from __future__ import annotations

import argparse
import csv
import hashlib
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs" / "DATA_INVENTORY.csv"
FETCH_MANIFEST = ROOT / "docs" / "DATA_FETCH.csv"
DEFAULT_DATA_DIR = ROOT / "data"
SUPPORTED_MODES = {"download", "build", "repository", "manual", "excluded"}
BUILD_PRIORITIES = {
    "uv run python -m scripts.prepare_public_data": 10,
    "uv run python -m builders.fetch_wikipedia_squad_club_cohesion": 20,
    "uv run python -m builders.build_in_tournament_adjustments": 30,
    "uv run python -m builders.build_in_tournament_shape_profiles": 40,
    "uv run python -m builders.build_style_matchup_edges": 50,
    "uv run python -m builders.analyze_world_cup_heat_latitude": 60,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_manifest() -> list[dict[str, str]]:
    inventory_rows = load_csv(INVENTORY)
    fetch_rows = load_csv(FETCH_MANIFEST)
    inventory = {row["file"]: row for row in inventory_rows}
    fetch = {row["file"]: row for row in fetch_rows}

    if len(inventory) != len(inventory_rows):
        raise RuntimeError("duplicate file in DATA_INVENTORY.csv")
    if len(fetch) != len(fetch_rows):
        raise RuntimeError("duplicate file in DATA_FETCH.csv")
    if inventory.keys() != fetch.keys():
        missing_fetch = sorted(inventory.keys() - fetch.keys())
        missing_inventory = sorted(fetch.keys() - inventory.keys())
        raise RuntimeError(
            "manifest file mismatch: "
            f"missing fetch rows={missing_fetch}; missing inventory rows={missing_inventory}"
        )

    rows: list[dict[str, str]] = []
    for filename in inventory:
        row = {**inventory[filename], **fetch[filename]}
        if row["mode"] not in SUPPORTED_MODES:
            raise RuntimeError(f"unsupported mode for {filename}: {row['mode']}")
        if row["mode"] == "download" and not row["source_url"]:
            raise RuntimeError(f"download URL missing for {filename}")
        if row["mode"] == "build" and not row["command"]:
            raise RuntimeError(f"build command missing for {filename}")
        rows.append(row)
    return rows


def matches_inventory(path: Path, row: dict[str, str]) -> bool:
    return path.stat().st_size == int(row["size_bytes"]) and sha256(path) == row["sha256"]


def download(row: dict[str, str], data_dir: Path, overwrite: bool) -> None:
    destination = data_dir / row["file"]
    if destination.exists() and matches_inventory(destination, row):
        print(f"OK       {row['file']}")
        return
    if destination.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite mismatched file: {destination}")

    data_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    request = urllib.request.Request(
        row["source_url"],
        headers={"User-Agent": "world-cup-prediction-model-data-fetch/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if not matches_inventory(temporary, row):
            raise RuntimeError(f"downloaded file failed checksum verification: {row['file']}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"FETCHED  {row['file']}")


def run_builds(rows: list[dict[str, str]], data_dir: Path) -> None:
    commands = list(dict.fromkeys(row["command"] for row in rows if row["mode"] == "build"))
    commands.sort(key=lambda command: BUILD_PRIORITIES.get(command, 100))
    for command in commands:
        print(f"BUILD    {command}")
        subprocess.run(shlex.split(command), cwd=ROOT, check=True)
    for row in rows:
        path = data_dir / row["file"]
        if not path.exists():
            raise RuntimeError(f"build did not create expected file: {path}")
        if path.stat().st_size == 0:
            raise RuntimeError(f"build created an empty file: {path}")
        print(f"BUILT    {row['file']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire data using the reviewed data manifests.")
    parser.add_argument("--list", action="store_true", help="List acquisition modes without changing files.")
    parser.add_argument("--file", action="append", default=[], help="Acquire only this file; repeat as needed.")
    parser.add_argument("--build", action="store_true", help="Run build commands for selected build-mode files.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a mismatched downloaded file.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Destination directory for downloads.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_manifest()
    selected_names = set(args.file)
    unknown = sorted(selected_names - {row["file"] for row in rows})
    if unknown:
        raise RuntimeError(f"unknown data files: {unknown}")
    selected = [row for row in rows if not selected_names or row["file"] in selected_names]

    if args.list:
        for row in selected:
            print(f"{row['mode']:<10} {row['file']}")
        return 0

    downloads = [row for row in selected if row["mode"] == "download"]
    data_dir = args.data_dir.resolve()
    for row in downloads:
        download(row, data_dir, args.overwrite)

    build_rows = [row for row in selected if row["mode"] == "build"]
    if args.build:
        if data_dir != DEFAULT_DATA_DIR.resolve():
            raise RuntimeError("--build only supports the repository data/ directory")
        run_builds(build_rows, data_dir)

    unavailable = [row for row in selected if row["mode"] in {"manual", "excluded"}]
    repository = [row for row in selected if row["mode"] == "repository"]
    repository_failures: list[str] = []
    for row in repository:
        path = data_dir / row["file"]
        status = "OK" if path.exists() and matches_inventory(path, row) else "MISSING"
        print(f"{status:<8} {row['file']} (repository)")
        if status == "MISSING":
            repository_failures.append(row["file"])
    for row in unavailable:
        print(f"SKIP     {row['file']} ({row['mode']}): {row['notes']}")
    if build_rows and not args.build:
        print(f"SKIP     {len(build_rows)} build-mode files; pass --build to run their commands.")

    if repository_failures:
        print(f"ERROR: missing or mismatched repository data: {repository_failures}", file=sys.stderr)
        return 1
    if selected_names and unavailable:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
