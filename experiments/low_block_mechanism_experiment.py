from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import realtime_context_adjusted_plan as realtime
from evaluation.evaluate_plan_against_results import evaluate
from predict_fifa_profile import expected_total_goals_value as base_expected_total_goals_value


OUTPUT_DIR = ROOT / "experiments" / "output"
SUMMARY_CSV = OUTPUT_DIR / "low_block_mechanism_summary.csv"
DETAIL_CSV = OUTPUT_DIR / "low_block_mechanism_details.csv"
LOW_LABELS = {"low_block", "low_event_favorite", "low_event"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["date_bjt"], row["time_bjt"], row["team_a"], row["team_b"]


def strip_low_labels(shape_labels: str) -> str:
    return ";".join(label for label in shape_labels.split(";") if label and label not in LOW_LABELS)


def scaled_low_expected_total(scale: float):
    def expected_total(
        lambda_a,
        lambda_b,
        p_a,
        p_draw,
        p_b,
        ranking_a=None,
        ranking_b=None,
        profile_a=None,
        profile_b=None,
        baselines=None,
        shape_labels="",
    ):
        full = base_expected_total_goals_value(
            lambda_a,
            lambda_b,
            p_a,
            p_draw,
            p_b,
            ranking_a,
            ranking_b,
            profile_a,
            profile_b,
            baselines,
            shape_labels,
        )
        if not ({label for label in shape_labels.split(";") if label} & LOW_LABELS):
            return full
        without_low = base_expected_total_goals_value(
            lambda_a,
            lambda_b,
            p_a,
            p_draw,
            p_b,
            ranking_a,
            ranking_b,
            profile_a,
            profile_b,
            baselines,
            strip_low_labels(shape_labels),
        )
        return without_low + (full - without_low) * scale

    return expected_total


def make_rows() -> list[dict[str, str]]:
    contexts = realtime.load_context()
    shapes = realtime.load_match_shapes()
    team_shape_profiles = realtime.load_team_shape_profiles()
    key_player_signals = realtime.load_key_player_signals()
    key_player_statuses = realtime.load_key_player_match_statuses()
    team_market_values = realtime.load_market_values()
    completed_matches = realtime.load_completed_matches()
    return [
        realtime.apply_context(
            row,
            contexts,
            shapes,
            team_shape_profiles,
            key_player_signals,
            key_player_statuses,
            team_market_values,
            completed_matches,
        )
        for row in realtime.load_predictions()
    ]


def disable_strong_favorite_protection(selected_bucket, total_buckets, lambda_a, lambda_b, p_a, p_b):
    return selected_bucket, total_buckets


def run_variant(
    name: str,
    low_expected_scale: float | None = None,
    xg_draw_low_event_factor: float | None = None,
    disable_protection: bool = False,
) -> tuple[Path, dict[str, float | int]]:
    original_expected = realtime.expected_total_goals_value
    original_protection = realtime.protect_strong_favorite_from_low_bucket
    original_xg_draw_low_event_factor = realtime.XG_DRAW_LOW_EVENT_FACTOR

    if low_expected_scale is not None:
        realtime.expected_total_goals_value = scaled_low_expected_total(low_expected_scale)
    if xg_draw_low_event_factor is not None:
        realtime.XG_DRAW_LOW_EVENT_FACTOR = xg_draw_low_event_factor
    if disable_protection:
        realtime.protect_strong_favorite_from_low_bucket = disable_strong_favorite_protection

    try:
        rows = make_rows()
    finally:
        realtime.expected_total_goals_value = original_expected
        realtime.protect_strong_favorite_from_low_bucket = original_protection
        realtime.XG_DRAW_LOW_EVENT_FACTOR = original_xg_draw_low_event_factor

    path = OUTPUT_DIR / f"low_block_mechanism_{name}.csv"
    write_csv(path, rows)
    return path, evaluate(path)


def metric_delta(metrics: dict, baseline: dict, key: str) -> str:
    value = metrics[key] - baseline[key]
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    baseline_path, baseline_metrics = run_variant("current")
    variants = [
        ("no_protection", None, None, True),
        ("no_low_expected", 0.0, None, True),
        ("half_low_expected", 0.5, None, True),
        ("soft_low_draw", None, 1.15, True),
        ("half_low_expected_soft_low_draw", 0.5, 1.15, True),
    ]
    summary_rows: list[dict[str, str]] = []
    detail_rows: list[dict[str, str]] = []
    baseline_rows = {row_key(row): row for row in read_csv(baseline_path)}

    for name, low_scale, draw_factor, no_protection in variants:
        path, metrics = run_variant(name, low_scale, draw_factor, no_protection)
        rows = read_csv(path)
        changed = [
            row
            for row in rows
            if baseline_rows[row_key(row)]["adjusted_total_goal_bucket"] != row["adjusted_total_goal_bucket"]
            or baseline_rows[row_key(row)]["adjusted_score_1_model"] != row["adjusted_score_1_model"]
            or baseline_rows[row_key(row)]["predicted_outcome"] != row["predicted_outcome"]
        ]
        summary_rows.append(
            {
                "variant": name,
                "changed_count": str(len(changed)),
                "changed_matches": "; ".join(f"{row['team_a']} vs {row['team_b']}" for row in changed[:12]),
                "outcome_hit": str(metrics["outcome_hit"]),
                "outcome_delta": metric_delta(metrics, baseline_metrics, "outcome_hit"),
                "top1_hit": str(metrics["top1_hit"]),
                "top1_delta": metric_delta(metrics, baseline_metrics, "top1_hit"),
                "top2_hit": str(metrics["top2_hit"]),
                "top2_delta": metric_delta(metrics, baseline_metrics, "top2_hit"),
                "any_exact": str(metrics["any_exact"]),
                "any_exact_delta": metric_delta(metrics, baseline_metrics, "any_exact"),
                "any_score_bucket": str(metrics["any_score_bucket"]),
                "any_score_bucket_delta": metric_delta(metrics, baseline_metrics, "any_score_bucket"),
                "mean_deviation": f"{metrics['mean_deviation']:.6f}",
                "mean_deviation_delta": metric_delta(metrics, baseline_metrics, "mean_deviation"),
                "median_deviation": f"{metrics['median_deviation']:.6f}",
                "median_deviation_delta": metric_delta(metrics, baseline_metrics, "median_deviation"),
            }
        )
        for row in changed:
            before = baseline_rows[row_key(row)]
            detail_rows.append(
                {
                    "variant": name,
                    "match": f"{row['team_a']} vs {row['team_b']}",
                    "shape_labels": row["shape_labels"],
                    "old_outcome": before["predicted_outcome"],
                    "new_outcome": row["predicted_outcome"],
                    "old_bucket": before["adjusted_total_goal_bucket"],
                    "new_bucket": row["adjusted_total_goal_bucket"],
                    "old_model": before["adjusted_score_1_model"],
                    "new_model": row["adjusted_score_1_model"],
                    "old_top2": before["adjusted_total_goals_top2"],
                    "new_top2": row["adjusted_total_goals_top2"],
                }
            )

    write_csv(SUMMARY_CSV, summary_rows)
    if detail_rows:
        write_csv(DETAIL_CSV, detail_rows)

    print(f"Summary: {SUMMARY_CSV}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
