from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.prepare_realtime_context_package import validate_and_convert

ROOT = Path(__file__).resolve().parents[1]


def payload() -> dict:
    source = {"label": "official", "url": "https://example.com/team-news", "published_at": None}
    multipliers = {
        "home_adaptation_multiplier": None,
        "travel_multiplier": 0.98,
        "weather_multiplier": 1.0,
        "cohesion_multiplier": None,
        "injury_multiplier": 0.97,
        "opponent_attack_multiplier": 1.02,
        "tempo_multiplier": 0.99,
    }
    return {
        "collector": {
            "model": "GPT-5.5",
            "reasoning_effort": "very_high",
            "collected_at_utc": "2026-06-10T00:00:00+00:00",
        },
        "match": {
            "date_bjt": "2026-06-12",
            "time_bjt": "00:00",
            "team_a": "Mexico",
            "team_b": "South Africa",
            "venue": "Mexico City",
        },
        "teams": [
            {"team": "Mexico", "multipliers": multipliers, "weather_high_c": 28, "travel_km": 0, "analysis_notes": "Reviewed evidence.", "sources": [source]},
            {"team": "South Africa", "multipliers": multipliers, "weather_high_c": 28, "travel_km": 14000, "analysis_notes": "Reviewed evidence.", "sources": [source]},
        ],
        "shape": {
            "pre_match_shapes": ["controlled_favorite"],
            "draw_multiplier": 0.98,
            "tempo_multiplier": 0.97,
            "favorite_attack_multiplier": 1.0,
            "underdog_attack_multiplier": 1.0,
            "notes": "Likely controlled match.",
            "sources": [source],
        },
        "key_players": [],
    }


def test_context_package_is_converted_to_model_inputs(tmp_path: Path) -> None:
    validate_and_convert(payload(), tmp_path)
    with (tmp_path / "realtime_team_context.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["team"] for row in rows] == ["Mexico", "South Africa"]
    assert rows[0]["travel_multiplier"] == "0.98"
    assert (tmp_path / "match_shape_context.csv").exists()
    assert (tmp_path / "world_cup_2026_key_player_match_status.csv").exists()
    assert (tmp_path / "context_package.json").exists()


def test_context_package_rejects_post_kickoff_collection(tmp_path: Path) -> None:
    value = payload()
    value["collector"]["collected_at_utc"] = "2026-06-11T16:00:00+00:00"
    with pytest.raises(ValueError, match="before kickoff"):
        validate_and_convert(value, tmp_path)


def test_context_package_rejects_out_of_range_multiplier(tmp_path: Path) -> None:
    value = payload()
    value["teams"][0]["multipliers"]["injury_multiplier"] = 0.5
    with pytest.raises(ValueError, match="outside"):
        validate_and_convert(value, tmp_path)


def test_documented_context_example_is_runnable(tmp_path: Path) -> None:
    import json

    value = json.loads(
        (ROOT / "examples" / "realtime_context_example.json").read_text(encoding="utf-8")
    )
    validate_and_convert(value, tmp_path)
    assert (tmp_path / "realtime_team_context.csv").exists()
