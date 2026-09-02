from __future__ import annotations

import csv
from pathlib import Path

import pytest

from evaluation import evaluate_finished_from_realtime_cache as evaluation
from evaluation.compare_base_and_realtime import comparison_rows


def test_empty_cache_does_not_write_evaluation_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    writes: list[str] = []
    monkeypatch.setattr(evaluation, "load_results", lambda: [{"match": "test"}])
    monkeypatch.setattr(evaluation, "load_cache_runs", lambda _cache_dir: [])
    monkeypatch.setattr(evaluation, "cache_for_match", lambda _result, _runs: None)
    monkeypatch.setattr(evaluation, "write_detail", lambda _rows: writes.append("detail"))
    monkeypatch.setattr(
        evaluation,
        "write_summary",
        lambda _rows, _skipped, _source: writes.append("summary"),
    )
    monkeypatch.setattr(
        evaluation,
        "write_bucket_reweight_experiment",
        lambda _rows: writes.append("experiment"),
    )

    with pytest.raises(RuntimeError, match="no pre-match predictions"):
        evaluation.main(source="cache")
    assert writes == []


def test_public_archive_rebuilds_published_match_rows() -> None:
    archived = evaluation.load_prediction_archive()
    rebuilt = []
    for result in evaluation.load_results():
        found = evaluation.archive_for_match(result, archived)
        if found is not None:
            rebuilt.append(evaluation.evaluated_row(result, *found))

    with evaluation.DETAIL_CSV.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        published = list(reader)
        fields = reader.fieldnames or []

    assert len(rebuilt) == 79
    assert [{field: row.get(field, "") for field in fields} for row in rebuilt] == published


def test_archive_rejects_post_kickoff_prediction(tmp_path: Path) -> None:
    result = evaluation.load_results()[0]
    prediction = {
        "date_bjt": result["date_bjt"],
        "time_bjt": result["time_bjt"],
        "team_a": result["team_a"],
        "team_b": result["team_b"],
    }
    archived = {
        evaluation.match_key(result): (
            {"run_id": "late", "created_at_utc": result["kickoff_utc"]},
            prediction,
        )
    }
    with pytest.raises(RuntimeError, match="not pre-match"):
        evaluation.archive_for_match(result, archived)


def test_base_and_realtime_evaluations_use_same_matches() -> None:
    base_rows, realtime_rows = comparison_rows()
    assert len(base_rows) == len(realtime_rows) == 79
    assert sum(evaluation.bool_field(row, "outcome_hit") for row in base_rows) == 54
    assert sum(evaluation.bool_field(row, "outcome_hit") for row in realtime_rows) == 53
    assert sum(evaluation.bool_field(row, "top1_bucket_hit") for row in base_rows) == 17
    assert sum(evaluation.bool_field(row, "top1_bucket_hit") for row in realtime_rows) == 26
