from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from evaluation.evaluate_finished_from_realtime_cache import (
    ARCHIVE_CSV,
    CACHE_DIR,
    cache_for_match,
    load_cache_runs,
    load_plan_rows,
    load_results,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_JSON = ROOT / "data" / "strict_pre_match_predictions_manifest.json"
REALTIME_COLLECTOR_MODEL = "GPT-5.5"
REALTIME_REASONING_EFFORT = "very_high"
ARCHIVE_FIELDS = [
    "cache_run_id",
    "cache_created_at_utc",
    "source_manifest_sha256",
    "source_plan_sha256",
    "realtime_collector_model",
    "realtime_reasoning_effort",
    "date_bjt",
    "time_bjt",
    "group",
    "team_a",
    "team_b",
    "predicted_outcome",
    "adjusted_total_goal_bucket",
    "adjusted_total_goals_top2",
    "adjusted_score_1_model",
    "adjusted_score_2_aggressive_prediction",
    "adjusted_score_3_market_value",
    "adjusted_score_4_upset",
    "bucket_primary_score",
    "aggressive_score",
    "market_value_score",
    "upset_score",
    "base_predicted_outcome",
    "base_p_a",
    "base_p_draw",
    "base_p_b",
    "base_xg_a",
    "base_xg_b",
    "base_selected_total_goal_bucket",
    "base_top_total_goal_buckets",
    "base_recommended_score",
    "base_aggressive_score",
    "base_market_value_score",
    "context_applied",
    "shape_applied",
    "shape_labels",
    "shape_notes",
    "group_round",
    "draw_acceptance_a",
    "draw_acceptance_b",
    "group_draw_multiplier",
    "group_tempo_multiplier",
    "adjusted_p_a",
    "adjusted_p_draw",
    "adjusted_p_b",
    "adjusted_xg_a",
    "adjusted_xg_b",
    "context_chain_multipliers",
    "context_signal_tags",
    "source_confidence_a",
    "source_confidence_b",
    "lineup_certainty_a",
    "lineup_certainty_b",
    "context_notes",
    "context_sources",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the minimum project-authored inputs needed to reproduce strict evaluation."
    )
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output", type=Path, default=ARCHIVE_CSV)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_JSON)
    parser.add_argument("--expected-matches", type=int)
    return parser.parse_args()


def export_archive(cache_dir: Path, output: Path, manifest_path: Path, expected_matches: int | None) -> int:
    results = load_results()
    runs = load_cache_runs(cache_dir)
    rows: list[dict[str, str]] = []
    source_runs: dict[str, dict[str, str]] = {}
    for result in results:
        found = cache_for_match(result, runs)
        if found is None:
            continue
        run, prediction = found
        base_path = run["base_prediction_csv"]
        if base_path is None:
            raise RuntimeError(f"base prediction snapshot missing for run {run['run_id']}")
        base_prediction = load_plan_rows(base_path).get(
            (result["date_bjt"], result["time_bjt"], result["team_a"], result["team_b"])
        )
        if base_prediction is None:
            raise RuntimeError(
                f"base prediction row missing: {result['team_a']} vs {result['team_b']}"
            )
        manifest_hash = sha256(run["manifest_path"])
        plan_hash = sha256(run["plan_csv"])
        source_runs[run["run_id"]] = {
            "run_id": run["run_id"],
            "created_at_utc": run["created_at_utc"].isoformat(),
            "manifest_sha256": manifest_hash,
            "plan_sha256": plan_hash,
        }
        row = {field: prediction.get(field, "") for field in ARCHIVE_FIELDS}
        row.update(
            {
                "cache_run_id": run["run_id"],
                "cache_created_at_utc": run["created_at_utc"].isoformat(),
                "source_manifest_sha256": manifest_hash,
                "source_plan_sha256": plan_hash,
                "realtime_collector_model": REALTIME_COLLECTOR_MODEL,
                "realtime_reasoning_effort": REALTIME_REASONING_EFFORT,
                "date_bjt": result["date_bjt"],
                "time_bjt": result["time_bjt"],
                "group": result["group"],
                "team_a": result["team_a"],
                "team_b": result["team_b"],
                "base_predicted_outcome": base_prediction["predicted_outcome"],
                "base_p_a": base_prediction["p_a"],
                "base_p_draw": base_prediction["p_draw"],
                "base_p_b": base_prediction["p_b"],
                "base_xg_a": base_prediction["xg_a"],
                "base_xg_b": base_prediction["xg_b"],
                "base_selected_total_goal_bucket": base_prediction[
                    "selected_total_goal_bucket"
                ],
                "base_top_total_goal_buckets": base_prediction["top_total_goal_buckets"],
                "base_recommended_score": base_prediction["recommended_score"],
                "base_aggressive_score": base_prediction["aggressive_score"],
                "base_market_value_score": base_prediction["market_value_score"],
            }
        )
        rows.append(row)

    if not rows:
        raise RuntimeError(f"no strict pre-match predictions found in {cache_dir}")
    if expected_matches is not None and len(rows) != expected_matches:
        raise RuntimeError(f"expected {expected_matches} matches, found {len(rows)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ARCHIVE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": 1,
        "description": "Minimal project-authored pre-kickoff prediction archive for strict evaluation.",
        "selection_rule": "Latest cached prediction with cache_created_at_utc strictly before kickoff.",
        "contains_results": False,
        "realtime_collection": {
            "model": REALTIME_COLLECTOR_MODEL,
            "reasoning_effort": REALTIME_REASONING_EFFORT,
            "method": "Human-initiated web research with LLM evidence synthesis and structured parameter entry."
        },
        "match_count": len(rows),
        "source_run_count": len(source_runs),
        "archive_file": output.name,
        "archive_sha256": sha256(output),
        "source_cache_latest_created_at_utc": max(
            item["created_at_utc"] for item in source_runs.values()
        ),
        "source_runs": sorted(source_runs.values(), key=lambda item: item["created_at_utc"]),
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return len(rows)


def main() -> int:
    args = parse_args()
    count = export_archive(args.cache_dir, args.output, args.manifest, args.expected_matches)
    print(f"Archive: {args.output}")
    print(f"Manifest: {args.manifest}")
    print(f"Matches: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
