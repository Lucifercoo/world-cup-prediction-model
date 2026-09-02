from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))

import wc_model
PROHIBITED_PATHS = (
    ROOT / "fetch_fifa_official_rankings.py",
    ROOT / "fetch_transfermarkt_world_cup_values.py",
    DATA_DIR / "fifa_rankings_history_datofutbol.csv",
)
PREDICTION_ARCHIVE = DATA_DIR / "strict_pre_match_predictions.csv"
PREDICTION_ARCHIVE_MANIFEST = DATA_DIR / "strict_pre_match_predictions_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    present = [str(path.relative_to(ROOT)) for path in PROHIBITED_PATHS if path.exists()]
    if present:
        raise RuntimeError(f"restricted legacy files are present: {present}")

    wc_model.require_model_inputs()
    annual_rows = read_rows("fifa_rankings_annual_start.csv")
    dates_2026 = {row["snapshot_date"] for row in annual_rows if row["year"] == "2026"}
    if dates_2026 != {"2026-06-11"}:
        raise RuntimeError(f"public mode uses an invalid 2026 FIFA snapshot: {sorted(dates_2026)}")

    proxy_rows = read_rows("transfermarkt_world_cup_2026_values.csv")
    if len(proxy_rows) != 48 or any("NOT Transfermarkt" not in row["source"] for row in proxy_rows):
        raise RuntimeError("public squad-value input is not the documented 48-team proxy")

    if read_rows("world_cup_2026_key_player_signals.csv"):
        raise RuntimeError("public mode must leave the optional key-player signal layer disabled")

    archive_rows = read_rows(PREDICTION_ARCHIVE.name)
    archive_manifest = json.loads(PREDICTION_ARCHIVE_MANIFEST.read_text(encoding="utf-8"))
    if len(archive_rows) != archive_manifest["match_count"]:
        raise RuntimeError("strict prediction archive row count does not match its manifest")
    if sha256(PREDICTION_ARCHIVE) != archive_manifest["archive_sha256"]:
        raise RuntimeError("strict prediction archive hash does not match its manifest")
    collection = archive_manifest.get("realtime_collection", {})
    if collection.get("model") != "GPT-5.5" or collection.get("reasoning_effort") != "very_high":
        raise RuntimeError("historical realtime collector provenance is missing or changed")
    if any(
        row.get("realtime_collector_model") != "GPT-5.5"
        or row.get("realtime_reasoning_effort") != "very_high"
        for row in archive_rows
    ):
        raise RuntimeError("prediction archive collector provenance does not match its manifest")
    forbidden_fields = {"actual_score", "outcome_hit", "top1_bucket_hit", "top2_bucket_hit"}
    if forbidden_fields.intersection(archive_rows[0]):
        raise RuntimeError("strict prediction archive contains result or evaluation fields")

    print("Public inputs and the strict pre-match evaluation archive are reproducible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
