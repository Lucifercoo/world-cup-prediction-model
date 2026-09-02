from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.apply_realtime_context_package import apply_package
from scripts.inspect_predictions import render, select_rows
from scripts.prepare_realtime_context_package import validate_and_convert


def test_context_package_upserts_without_duplicate_rows(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "examples" / "realtime_context_example.json").read_text(encoding="utf-8")
    )
    package_dir = tmp_path / "package"
    data_dir = tmp_path / "data"
    validate_and_convert(payload, package_dir)

    first = apply_package(package_dir, data_dir)
    second = apply_package(package_dir, data_dir)

    assert all(inserted >= 0 and replaced == 0 for _, inserted, replaced in first)
    assert all(inserted == 0 for _, inserted, _ in second)
    for name, expected_rows in (("realtime_team_context.csv", 2), ("match_shape_context.csv", 1)):
        with (data_dir / name).open(encoding="utf-8-sig", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == expected_rows


def test_context_apply_revalidates_source_json(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "context_package.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="root fields differ"):
        apply_package(package_dir, tmp_path / "data")


def test_prediction_inspector_filters_and_renders(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    fields = [
        "date_bjt",
        "time_bjt",
        "team_a",
        "team_b",
        "predicted_outcome",
        "adjusted_total_goal_bucket",
        "backup_total_goal_bucket",
        "adjusted_score_1_model",
        "adjusted_score_2_aggressive_prediction",
        "adjusted_score_3_market_value",
        "adjusted_score_4_upset",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "date_bjt": "2026-07-20",
                "time_bjt": "03:00",
                "team_a": "Spain",
                "team_b": "Argentina",
                "predicted_outcome": "D",
                "adjusted_total_goal_bucket": "2-3 goals",
                "backup_total_goal_bucket": "0-1 goals",
                "adjusted_score_1_model": "1-1",
                "adjusted_score_2_aggressive_prediction": "0-0",
                "adjusted_score_3_market_value": "2-1",
                "adjusted_score_4_upset": "0-1",
            }
        )

    rows = select_rows(path, "2026-07-20", "spain")
    output = render(rows)

    assert len(rows) == 1
    assert "Spain vs Argentina" in output
    assert "Draw" in output
