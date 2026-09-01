from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import realtime_context_adjusted_plan as realtime
from evaluation.evaluate_plan_against_results import evaluate


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SUMMARY_MD = OUTPUT_DIR / "low_block_effect_experiment_summary.md"
DETAIL_CSV = OUTPUT_DIR / "low_block_effect_experiment_details.csv"
LOW_LABELS = {"low_block", "low_event_favorite", "low_event"}
FIELDS = [
    "date_bjt",
    "time_bjt",
    "group",
    "team_a",
    "team_b",
    "venue",
    "predicted_outcome",
    "p_a",
    "p_draw",
    "p_b",
    "adjusted_p_a",
    "adjusted_p_draw",
    "adjusted_p_b",
    "xg_a",
    "xg_b",
    "adjusted_xg_a",
    "adjusted_xg_b",
    "selected_total_goal_bucket",
    "adjusted_total_goal_bucket",
    "adjusted_total_goals_top2",
    "bucket_primary_score",
    "adjusted_score_1_model",
    "bucket_complement_score",
    "aggressive_score",
    "adjusted_score_2_aggressive_prediction",
    "market_value_raw_score",
    "market_value_score",
    "adjusted_score_3_market_value",
    "upset_score",
    "upset_score_probability",
    "adjusted_score_4_upset",
    "adjusted_score_4_upset_probability",
    "xg_goal_diff",
    "xg_outcome_edge",
    "legacy_outcome_edge",
    "outcome_edge_conflict",
    "risk_label",
    "risk_reasons",
    "context_applied",
    "shape_applied",
    "shape_labels",
    "shape_notes",
    "group_round",
    "draw_acceptance_a",
    "draw_acceptance_b",
    "group_draw_multiplier",
    "group_tempo_multiplier",
    "group_context_complete",
    "group_context_notes",
    "context_chain_multipliers",
    "context_signal_tags",
    "source_confidence_a",
    "source_confidence_b",
    "lineup_certainty_a",
    "lineup_certainty_b",
    "context_notes",
    "context_sources",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def remove_low_labels(shape_labels: str) -> str:
    labels = [label for label in shape_labels.split(";") if label and label not in LOW_LABELS]
    return ";".join(labels)


def make_rows() -> list[dict[str, str]]:
    contexts = realtime.load_context()
    shapes = realtime.load_match_shapes()
    key_player_signals = realtime.load_key_player_signals()
    key_player_statuses = realtime.load_key_player_match_statuses()
    team_market_values = realtime.load_market_values()
    completed_matches = realtime.load_completed_matches()
    return [
        realtime.apply_context(
            row,
            contexts,
            shapes,
            key_player_signals,
            key_player_statuses,
            team_market_values,
            completed_matches,
        )
        for row in realtime.load_predictions()
    ]


def run_variant(name: str, patch_kind: str) -> tuple[Path, dict]:
    original_second = realtime.second_bucket_from_expected_total_goals
    original_expected = realtime.expected_total_goals_value

    if patch_kind in {"no_top2_bias", "no_both"}:
        def second_bucket(expected_total_goals: float, selected_bucket: str, shape_labels: str = "") -> str:
            return original_second(expected_total_goals, selected_bucket, remove_low_labels(shape_labels))

        realtime.second_bucket_from_expected_total_goals = second_bucket

    if patch_kind in {"no_expected_penalty", "no_both"}:
        def expected_total(
            lambda_a: float,
            lambda_b: float,
            p_a: float,
            p_draw: float,
            p_b: float,
            ranking_a=None,
            ranking_b=None,
            profile_a=None,
            profile_b=None,
            baselines=None,
            shape_labels: str = "",
        ) -> float:
            return original_expected(
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
                remove_low_labels(shape_labels),
            )

        realtime.expected_total_goals_value = expected_total

    try:
        rows = make_rows()
    finally:
        realtime.second_bucket_from_expected_total_goals = original_second
        realtime.expected_total_goals_value = original_expected

    path = OUTPUT_DIR / f"low_block_{name}.csv"
    write_csv(path, rows, FIELDS)
    return path, evaluate(path)


def metric_row(metrics: dict) -> str:
    return (
        f"{metrics['outcome_hit']}/{metrics['matches']} | "
        f"{metrics['top1_hit']}/{metrics['matches']} | "
        f"{metrics['top2_hit']}/{metrics['matches']} | "
        f"{metrics['any_exact']}/{metrics['matches']} | "
        f"{metrics['any_score_bucket']}/{metrics['matches']} | "
        f"{metrics['any_score_outcome']}/{metrics['matches']} | "
        f"{metrics['mean_deviation']:.3f} | "
        f"{metrics['median_deviation']:.3f}"
    )


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["date_bjt"], row["time_bjt"], row["team_a"], row["team_b"]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    baseline_path = realtime.ADJUSTED_CSV
    baseline_metrics = evaluate(baseline_path)
    variants = [
        ("no_top2_bias", "no_top2_bias"),
        ("no_expected_penalty", "no_expected_penalty"),
        ("no_both", "no_both"),
    ]
    outputs = [(name, *run_variant(name, kind)) for name, kind in variants]

    baseline_rows = {row_key(row): row for row in read_csv(baseline_path)}
    best_name, best_path, _ = max(
        outputs,
        key=lambda item: (
            item[2]["mean_deviation"] < baseline_metrics["mean_deviation"],
            item[2]["any_exact"] - baseline_metrics["any_exact"],
            item[2]["any_score_bucket"] - baseline_metrics["any_score_bucket"],
            baseline_metrics["mean_deviation"] - item[2]["mean_deviation"],
        ),
    )
    detail_rows = []
    for row in read_csv(best_path):
        before = baseline_rows[row_key(row)]
        if (
            row["adjusted_total_goals_top2"] == before["adjusted_total_goals_top2"]
            and row["adjusted_score_2_aggressive_prediction"] == before["adjusted_score_2_aggressive_prediction"]
            and row["adjusted_score_1_model"] == before["adjusted_score_1_model"]
        ):
            continue
        detail_rows.append(
            {
                "date_bjt": row["date_bjt"],
                "time_bjt": row["time_bjt"],
                "match": f"{row['team_a']} vs {row['team_b']}",
                "shape_labels": row["shape_labels"],
                "old_top2": before["adjusted_total_goals_top2"],
                "new_top2": row["adjusted_total_goals_top2"],
                "old_scores": " / ".join(
                    before[column]
                    for column in (
                        "adjusted_score_1_model",
                        "adjusted_score_2_aggressive_prediction",
                        "adjusted_score_3_market_value",
                        "adjusted_score_4_upset",
                    )
                ),
                "new_scores": " / ".join(
                    row[column]
                    for column in (
                        "adjusted_score_1_model",
                        "adjusted_score_2_aggressive_prediction",
                        "adjusted_score_3_market_value",
                        "adjusted_score_4_upset",
                    )
                ),
            }
        )
    if detail_rows:
        write_csv(DETAIL_CSV, detail_rows, list(detail_rows[0]))

    lines = [
        "# Low Block Effect Experiment",
        "",
        "| Variant | Outcome | Top1 | Top2 | Exact | Score bucket | Score outcome | Mean dev | Median dev |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| current | {metric_row(baseline_metrics)} |",
    ]
    for name, path, metrics in outputs:
        lines.append(f"| {name} | {metric_row(metrics)} |")
    lines.extend(["", f"Best detail basis: `{best_name}`", f"Detail CSV: `{DETAIL_CSV}`"])
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Summary: {SUMMARY_MD}")
    print(f"Current: {baseline_metrics}")
    for name, path, metrics in outputs:
        print(name, metrics, path)


if __name__ == "__main__":
    main()
