from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import wc_model


def test_required_inputs_fail_before_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wc_model, "DATA_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="fifa_rankings_annual_start.csv"):
        wc_model.require_model_inputs()


def test_header_only_restricted_inputs_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in wc_model.REQUIRED_MODEL_INPUTS:
        schema = wc_model.ROOT / "schemas" / name
        (tmp_path / name).write_text(schema.read_text(encoding="utf-8-sig"), encoding="utf-8")
    monkeypatch.setattr(wc_model, "DATA_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="contains no data rows"):
        wc_model.require_model_inputs()


def test_pipeline_order_and_replay_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(wc_model, "require_model_inputs", lambda: None)
    monkeypatch.setattr(
        wc_model,
        "run_module",
        lambda module, *arguments: calls.append((module, arguments)),
    )

    wc_model.run_pipeline(replay=True)

    assert [module for module, _ in calls] == [step.module for step in wc_model.PIPELINE_STEPS]
    assert [arguments for _, arguments in calls] == [
        ("--replay",) if step.module == "builders.build_in_tournament_adjustments" else ()
        for step in wc_model.PIPELINE_STEPS
    ]


def test_report_without_build_uses_cached_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        wc_model,
        "run_module",
        lambda module, *arguments: calls.append((module, arguments)),
    )

    assert wc_model.main(["report", "2026-07-20", "--no-build"]) == 0
    assert calls == [("reports.daily_match_report", ("2026-07-20", "--no-refresh"))]


def test_report_rejects_replay_without_build() -> None:
    with pytest.raises(RuntimeError, match="--replay cannot be used with --no-build"):
        wc_model.main(["report", "2026-07-20", "--no-build", "--replay"])


def test_data_arguments_are_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        wc_model,
        "run_module",
        lambda module, *arguments: calls.append((module, arguments)),
    )

    assert wc_model.main(["data", "--build", "--file", "style_matchup_edges.csv"]) == 0
    assert calls == [
        ("scripts.fetch_data", ("--build", "--file", "style_matchup_edges.csv")),
    ]


def test_registered_experiment_modules_exist() -> None:
    missing = [
        experiment.module
        for experiment in wc_model.EXPERIMENTS.values()
        if importlib.util.find_spec(experiment.module) is None
    ]
    assert missing == []
    registered_files = {
        f"{experiment.module.rsplit('.', maxsplit=1)[-1]}.py"
        for experiment in wc_model.EXPERIMENTS.values()
    }
    experiment_files = {
        path.name for path in (wc_model.ROOT / "experiments").glob("*.py")
    } - {"__init__.py"}
    assert registered_files == experiment_files


def test_experiment_list_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    assert wc_model.main(["experiment", "list"]) == 0
    output = capsys.readouterr().out
    assert "low-block-effect" in output
    assert "xg-score-generation" in output
