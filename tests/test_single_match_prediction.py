from __future__ import annotations

from argparse import Namespace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import predict_fifa_profile as model
import single_match_prediction as single
from predict import host_multiplier


def test_parse_kickoff_requires_timezone() -> None:
    with pytest.raises(Exception, match="include a timezone"):
        single.parse_kickoff("2026-09-03T20:00:00")


def test_chinese_team_name_and_home_team_are_supported() -> None:
    kickoff = single.parse_kickoff("2026-09-03T20:00:00+08:00")
    match = single.internal_match(
        single.resolve_team("阿根廷"),
        single.resolve_team("比利时"),
        kickoff,
        "friendly",
        "Brussels",
        "b",
    )

    assert match.team_a == "Argentina"
    assert match.team_b == "Belgium"
    assert match.home_team == "Belgium"
    assert model.host_multiplier("Belgium", match.venue, match.home_team) > 1.0
    assert model.host_multiplier("Argentina", match.venue, match.home_team) == 1.0


def test_explicit_neutral_venue_does_not_grant_legacy_us_host_bonus() -> None:
    kickoff = single.parse_kickoff("2026-09-03T20:00:00+08:00")
    match = single.internal_match(
        "United States",
        "Belgium",
        kickoff,
        "friendly",
        "未指定",
        "neutral",
    )

    assert match.home_team == ""
    assert host_multiplier("United States", match.venue, match.home_team) == 1.0


def test_missing_market_score_is_not_excluded_from_upset_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, set[tuple[int, int]]] = {}
    prediction = {
        "selected_total_goal_bucket": "2-3球",
        "top_total_goal_buckets": "2-3球 40%; 0-1球 30%",
        "p_a": 0.5,
        "p_draw": 0.3,
        "p_b": 0.2,
        "xg_a": 1.5,
        "xg_b": 0.8,
        "bucket_primary_score": "2-0",
        "bucket_complement_score": "1-0",
        "market_value_score": "1-1",
    }

    monkeypatch.setattr(single.model, "outcome_adjusted_scores", lambda *args: [(0, 0, 1.0)])
    def fake_select(cells, buckets, p_a, p_draw, p_b, excluded):
        captured["excluded"] = excluded
        return 0, 0, 1.0

    monkeypatch.setattr(single.model, "select_upset_or_compression_score", fake_select)

    adjusted = single.remove_unavailable_market_influence(prediction)

    assert captured["excluded"] == {(2, 0), (1, 0)}
    assert adjusted["adjusted_score_4_upset"] == "0-0"


def test_missing_optional_inputs_disable_their_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(single, "MARKET_VALUE_CSV", tmp_path / "missing-values.csv")
    monkeypatch.setattr(single, "COHESION_CSV", tmp_path / "missing-cohesion.csv")

    values, values_used, _ = single.optional_market_values(("Argentina", "Belgium"))
    cohesion, cohesion_used, _ = single.optional_cohesion(("Argentina", "Belgium"))

    assert values_used is False
    assert cohesion_used is False
    assert all(value.total_eur_m == 149.0 for value in values.values())
    assert all(value.multiplier == 1.0 for value in cohesion.values())


def test_prediction_runs_without_optional_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_csv = tmp_path / "profiles.csv"
    ranking_csv = tmp_path / "rankings.csv"
    profile_csv.touch()
    ranking_csv.touch()
    monkeypatch.setattr(single.model, "PROFILE_CSV", profile_csv)
    monkeypatch.setattr(single.model, "LIVE_RANKING_CSV", ranking_csv)
    monkeypatch.setattr(single.model, "FIFA_RANKING_CSV", tmp_path / "missing-fifa.csv")
    monkeypatch.setattr(single, "LIVE_RESULTS_CSV", tmp_path / "missing-results.csv")
    monkeypatch.setattr(single, "MARKET_VALUE_CSV", tmp_path / "missing-values.csv")
    monkeypatch.setattr(single, "COHESION_CSV", tmp_path / "missing-cohesion.csv")
    rankings = {
        team: SimpleNamespace(rank=rank, points=points, snapshot_date="live:2026-06-11")
        for team, rank, points in (
            ("Argentina", 1, 1877.0),
            ("Belgium", 8, 1760.0),
        )
    }
    profiles = {
        team: SimpleNamespace(style="攻守兼备型")
        for team in rankings
    }
    raw_prediction = {
        "predicted_outcome": "A",
        "p_a": 0.5,
        "p_draw": 0.3,
        "p_b": 0.2,
        "xg_a": 1.5,
        "xg_b": 0.8,
        "selected_total_goal_bucket": "2-3球",
        "top_total_goal_buckets": "2-3球 40%; 0-1球 30%",
        "bucket_primary_score": "2-0",
        "bucket_complement_score": "1-0",
        "market_value_score": "1-1",
        "upset_score": "0-0",
        "risk_label": "中",
        "risk_reasons": "测试",
    }
    monkeypatch.setattr(single.model, "load_fifa_rankings", lambda: rankings)
    monkeypatch.setattr(single.model, "load_profiles", lambda: profiles)
    monkeypatch.setattr(single.model, "profile_baselines", lambda values: object())
    monkeypatch.setattr(single.model, "predict_match", lambda *args: raw_prediction)
    args = Namespace(
        team_a="阿根廷",
        team_b="比利时",
        kickoff=single.parse_kickoff("2026-09-03T20:00:00+08:00"),
        stage="friendly",
        venue="布鲁塞尔",
        home="b",
    )

    result = single.build_prediction(args)

    assert result["mode"] == "核心预测"
    assert result["prediction"]["scores"]["身价"] == "不可用"
    assert result["prediction"]["scores"]["模型"] != "不可用"
    assert result["prediction"]["scores"]["备选"] != "不可用"
    assert result["prediction"]["scores"]["爆冷"] != "不可用"


def test_tournament_state_is_restored_after_single_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = {
        "Argentina": model.InTournamentAdjustment(
            effective_after_bjt=datetime(2026, 1, 1),
            points_adjustment=1.0,
            attack_multiplier=1.0,
            tempo_multiplier=1.0,
            reason="test",
        )
    }
    monkeypatch.setattr(model, "IN_TOURNAMENT_ADJUSTMENTS", marker)

    with single.without_tournament_adjustments():
        assert model.IN_TOURNAMENT_ADJUSTMENTS == {}

    assert model.IN_TOURNAMENT_ADJUSTMENTS is marker


def test_stage_rules_skip_matches_without_standings_and_apply_to_knockouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {"predicted_outcome": "A"}
    calls: list[dict] = []
    monkeypatch.setattr(
        single.realtime,
        "apply_context",
        lambda row, **kwargs: calls.append(kwargs) or {**row, "stage_applied": True},
    )

    assert single.apply_stage_rules(raw, "friendly", {}) is raw
    assert single.apply_stage_rules(raw, "group", {}) is raw
    adjusted = single.apply_stage_rules(raw, "qf", {})

    assert adjusted["stage_applied"] is True
    assert len(calls) == 1


def test_stale_status_rejects_data_after_kickoff() -> None:
    with pytest.raises(RuntimeError, match="later than kickoff"):
        single.stale_status(
            datetime(2026, 9, 4).date(),
            datetime(2026, 9, 3).date(),
        )
