from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import realtime_context_adjusted_plan as realtime_plan
from predict_fifa_profile import (
    score_outcome,
    total_goal_bucket,
    total_goal_bucket_from_expected,
)
from realtime_context_adjusted_plan import (
    TeamContext,
    apply_context,
)
from reports.realtime_output import (
    REALTIME_OUTPUT_FIELDS,
    write_csv,
    write_markdown,
    write_realtime_cache,
)
from style_matchups import StyleMatchupEffect


@pytest.fixture(autouse=True)
def isolate_generated_style_statistics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        realtime_plan,
        "style_matchup_effect",
        lambda _features_a, _features_b: StyleMatchupEffect(0.0, 1.0, ()),
    )


def base_prediction_row() -> dict[str, str]:
    return {
        "group": "R16",
        "date_bjt": "2026-06-12",
        "time_bjt": "00:00",
        "date_et": "2026-06-11",
        "time_et": "12:00",
        "team_a": "Alpha",
        "team_b": "Beta",
        "venue": "Test Venue",
        "fifa_rank_a": "12",
        "fifa_rank_b": "38",
        "style_a": "攻守兼备型",
        "style_b": "防守型",
        "style_features_a": "attack_high;style_complete",
        "style_features_b": "low_event;style_defensive",
        "xg_a": "2.20",
        "xg_b": "0.75",
        "market_value_raw_score": "3-0",
        "market_value_score": "3-0",
        "selected_total_goal_bucket": "2-3球",
        "recommended_score": "2-0",
        "risk_label": "一般",
        "risk_reasons": "",
        "p_a": "0.60",
        "p_draw": "0.25",
        "p_b": "0.15",
    }


def neutral_context() -> TeamContext:
    return TeamContext(
        home_adaptation_multiplier=1.0,
        travel_multiplier=1.0,
        weather_multiplier=1.0,
        cohesion_multiplier=1.0,
        injury_multiplier=1.0,
        attack_multiplier=1.0,
        opponent_attack_multiplier=1.0,
        tempo_multiplier=1.0,
        source_confidence_multiplier=1.0,
        lineup_certainty_multiplier=1.0,
        notes="",
        source_urls="",
        weather_high_c="",
        travel_km="",
        defense_leak_evidence=False,
        underdog_goal_evidence=False,
    )


def run_apply_context(*, with_context: bool) -> dict[str, str]:
    row = base_prediction_row()
    match_name = f"{row['team_a']} vs {row['team_b']}"
    contexts = (
        {
            (match_name, row["team_a"]): neutral_context(),
            (match_name, row["team_b"]): neutral_context(),
        }
        if with_context
        else {}
    )
    return apply_context(
        row,
        contexts,
        shapes={},
        team_shape_profiles={},
        key_player_signals={},
        key_player_statuses={},
        team_market_values={},
        completed_matches=[],
    )


def test_score_outcome_contract() -> None:
    assert score_outcome(2, 1) == "A"
    assert score_outcome(1, 1) == "D"
    assert score_outcome(0, 2) == "B"


def test_total_goal_bucket_boundaries() -> None:
    assert [total_goal_bucket(value) for value in range(9)] == [
        "0-1球",
        "0-1球",
        "2-3球",
        "2-3球",
        "4-5球",
        "4-5球",
        "6-8球",
        "6-8球",
        "6-8球",
    ]
    assert total_goal_bucket_from_expected(1.49) == "0-1球"
    assert total_goal_bucket_from_expected(1.50) == "2-3球"
    assert total_goal_bucket_from_expected(3.50) == "4-5球"
    assert total_goal_bucket_from_expected(5.50) == "6-8球"


def test_apply_context_without_team_context_contract() -> None:
    result = run_apply_context(with_context=False)
    assert result["context_applied"] == "FALSE"
    assert result["predicted_outcome"] == "A"
    assert result["adjusted_total_goal_bucket"] == "2-3球"
    assert result["backup_total_goal_bucket"] == "4-5球"
    assert result["adjusted_score_1_model"] == "2-1"
    assert result["adjusted_score_2_aggressive_prediction"] == "3-1"
    assert result["adjusted_score_3_market_value"] == "1-1"
    assert result["adjusted_score_4_upset"] == "0-0"
    assert result["adjusted_p_a"] == "0.616000"
    assert result["adjusted_p_draw"] == "0.230000"
    assert result["adjusted_p_b"] == "0.154000"
    assert result["adjusted_xg_a"] == "2.1560"
    assert result["adjusted_xg_b"] == "0.7350"


def test_apply_context_with_team_context_contract() -> None:
    result = run_apply_context(with_context=True)
    assert result["context_applied"] == "TRUE"
    assert result["predicted_outcome"] == "A"
    assert result["adjusted_total_goal_bucket"] == "2-3球"
    assert result["backup_total_goal_bucket"] == "4-5球"
    assert result["adjusted_score_1_model"] == "2-1"
    assert result["adjusted_score_2_aggressive_prediction"] == "3-1"
    assert result["adjusted_score_3_market_value"] == "1-1"
    assert result["adjusted_score_4_upset"] == "0-0"
    assert result["adjusted_p_a"] == "0.616000"
    assert result["adjusted_p_draw"] == "0.230000"
    assert result["adjusted_p_b"] == "0.154000"
    assert result["adjusted_xg_a"] == "2.1560"
    assert result["adjusted_xg_b"] == "0.7350"


def test_realtime_output_schema_contract() -> None:
    result = run_apply_context(with_context=True)
    assert len(REALTIME_OUTPUT_FIELDS) == len(set(REALTIME_OUTPUT_FIELDS))
    assert set(REALTIME_OUTPUT_FIELDS) <= result.keys()
    assert result["style_features_a"] == "attack_high;style_complete"
    assert result["style_features_b"] == "low_event;style_defensive"


def test_realtime_output_files_and_manifest(tmp_path: Path) -> None:
    result = run_apply_context(with_context=True)
    csv_path = tmp_path / "plan.csv"
    markdown_path = tmp_path / "plan.md"
    source_path = tmp_path / "source.csv"
    source_path.write_text("value\n1\n", encoding="utf-8")

    write_csv(csv_path, [result])
    write_markdown(markdown_path, [result])
    cache_path = Path(
        write_realtime_cache(
            [result],
            cache_dir=tmp_path / "cache",
            source_files=[("source", source_path)],
            output_files=[("csv", csv_path), ("markdown", markdown_path)],
            runtime_parameters={"mode": "test", "team_shape_profile_mode": "off"},
        )
    )

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        assert tuple(next(csv.reader(handle))) == REALTIME_OUTPUT_FIELDS
    assert "Alpha vs Beta" in markdown_path.read_text(encoding="utf-8")
    manifest = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["team_shape_profile_mode"] == "off"
    assert manifest["runtime_parameters"] == {"mode": "test", "team_shape_profile_mode": "off"}
    assert manifest["row_count"] == 1
    assert manifest["context_applied_count"] == 1
