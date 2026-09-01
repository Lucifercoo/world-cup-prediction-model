from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import predict_fifa_profile as profile_model
from evaluate_plan_against_results import evaluate


PLAN_CSV = ROOT / "output" / "realtime_context_adjusted_plan.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SUMMARY_MD = OUTPUT_DIR / "outcome_conflict_score_experiment_summary.md"
DETAIL_CSV = OUTPUT_DIR / "outcome_conflict_score_experiment_details.csv"
SCORE_COLUMNS = [
    "adjusted_score_1_model",
    "adjusted_score_2_aggressive_prediction",
    "adjusted_score_3_market_value",
    "adjusted_score_4_upset",
]


@dataclass(frozen=True)
class Variant:
    name: str
    model_conservative: bool
    backup_legacy: bool
    upset_conflict: bool
    same_direction_as_draw: bool


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_score(score: str) -> tuple[int, int]:
    home, away = score.split("-", maxsplit=1)
    return int(home), int(away)


def score_text(score: tuple[int, int]) -> str:
    return f"{score[0]}-{score[1]}"


def score_outcome(score: tuple[int, int]) -> str:
    if score[0] > score[1]:
        return "A"
    if score[1] > score[0]:
        return "B"
    return "D"


def legacy_top_outcome(row: dict[str, str]) -> str:
    probabilities = {
        "A": float(row["adjusted_p_a"]),
        "D": float(row["adjusted_p_draw"]),
        "B": float(row["adjusted_p_b"]),
    }
    return max(probabilities.items(), key=lambda item: item[1])[0]


def opposite_outcome(outcome: str) -> str:
    if outcome == "A":
        return "B"
    if outcome == "B":
        return "A"
    raise ValueError("draw has no opposite outcome")


def parse_bucket_order(row: dict[str, str]) -> list[str]:
    buckets = [row["adjusted_total_goal_bucket"]]
    for part in row["adjusted_total_goals_top2"].split(";"):
        piece = part.strip().split()
        if piece and piece[0] not in buckets:
            buckets.append(piece[0])
    for bucket in profile_model.TOTAL_GOAL_BUCKET_LABELS:
        if bucket not in buckets:
            buckets.append(bucket)
    return buckets


def score_bucket(score: tuple[int, int]) -> str:
    return profile_model.total_goal_bucket(score[0] + score[1])


def cells_for_row(row: dict[str, str]) -> list[tuple[int, int, float]]:
    return profile_model.outcome_adjusted_scores(
        float(row["adjusted_xg_a"]),
        float(row["adjusted_xg_b"]),
        float(row["adjusted_p_a"]),
        float(row["adjusted_p_draw"]),
        float(row["adjusted_p_b"]),
    )


def candidate_scores(
    cells: list[tuple[int, int, float]],
    buckets: set[str],
    outcomes: set[str],
    excluded: set[tuple[int, int]],
) -> list[tuple[int, int, float]]:
    return [
        cell
        for cell in cells
        if (cell[0], cell[1]) not in excluded
        and score_bucket((cell[0], cell[1])) in buckets
        and score_outcome((cell[0], cell[1])) in outcomes
    ]


def best_probability_score(
    cells: list[tuple[int, int, float]],
    buckets: set[str],
    outcomes: set[str],
    excluded: set[tuple[int, int]],
) -> tuple[int, int, float] | None:
    candidates = candidate_scores(cells, buckets, outcomes, excluded)
    if not candidates:
        return None
    return candidates[0]


def conservative_score(
    cells: list[tuple[int, int, float]],
    bucket: str,
    outcome: str,
    excluded: set[tuple[int, int]],
) -> tuple[int, int, float] | None:
    candidates = candidate_scores(cells, {bucket}, {outcome}, excluded)
    if not candidates:
        return None
    if outcome == "D":
        return candidates[0]
    return min(
        candidates,
        key=lambda cell: (
            abs(cell[0] - cell[1]),
            1 if min(cell[0], cell[1]) == 0 else 0,
            -(cell[0] + cell[1]),
            -cell[2],
        ),
    )


def low_event_score(
    cells: list[tuple[int, int, float]],
    buckets: set[str],
    outcomes: set[str],
    excluded: set[tuple[int, int]],
) -> tuple[int, int, float] | None:
    candidates = candidate_scores(cells, buckets, outcomes, excluded)
    if not candidates:
        return None
    return min(candidates, key=lambda cell: (cell[0] + cell[1], abs(cell[0] - cell[1]), -cell[2]))


def conflict_backup_outcome(row: dict[str, str], legacy: str, variant: Variant) -> str:
    predicted = row["predicted_outcome"]
    if legacy != predicted:
        return legacy
    if variant.same_direction_as_draw and predicted != "D":
        return "D"
    if predicted == "D":
        return "A" if float(row["adjusted_p_a"]) >= float(row["adjusted_p_b"]) else "B"
    return predicted


def conflict_upset_outcomes(row: dict[str, str], legacy: str) -> set[str]:
    predicted = row["predicted_outcome"]
    if predicted == "D":
        return {legacy} if legacy != "D" else {"A", "B"}
    if legacy != predicted:
        return {"D", legacy}
    return {"D", opposite_outcome(predicted)}


def apply_variant(row: dict[str, str], variant: Variant) -> dict[str, str]:
    updated = dict(row)
    if row.get("outcome_edge_conflict") != "TRUE":
        updated["conflict_score_variant"] = variant.name
        updated["conflict_score_notes"] = "not_conflict"
        return updated

    cells = cells_for_row(row)
    bucket_order = parse_bucket_order(row)
    selected_bucket = row["adjusted_total_goal_bucket"]
    second_bucket = next(bucket for bucket in bucket_order if bucket != selected_bucket)
    predicted = row["predicted_outcome"]
    legacy = legacy_top_outcome(row)
    excluded = {parse_score(row["adjusted_score_3_market_value"])}
    changed: list[str] = []

    if variant.model_conservative:
        new_model = conservative_score(cells, selected_bucket, predicted, excluded)
        if new_model is not None:
            updated["bucket_primary_score"] = score_text((new_model[0], new_model[1]))
            updated["adjusted_score_1_model"] = score_text((new_model[0], new_model[1]))
            excluded.add((new_model[0], new_model[1]))
            changed.append("model_conservative")
        else:
            excluded.add(parse_score(updated["adjusted_score_1_model"]))
    else:
        excluded.add(parse_score(updated["adjusted_score_1_model"]))

    if variant.backup_legacy:
        target = conflict_backup_outcome(row, legacy, variant)
        new_backup = best_probability_score(cells, {second_bucket, selected_bucket}, {target}, excluded)
        if new_backup is not None:
            updated["bucket_complement_score"] = score_text((new_backup[0], new_backup[1]))
            updated["aggressive_score"] = score_text((new_backup[0], new_backup[1]))
            updated["adjusted_score_2_aggressive_prediction"] = score_text((new_backup[0], new_backup[1]))
            excluded.add((new_backup[0], new_backup[1]))
            changed.append(f"backup_{target}")
        else:
            excluded.add(parse_score(updated["adjusted_score_2_aggressive_prediction"]))
    else:
        excluded.add(parse_score(updated["adjusted_score_2_aggressive_prediction"]))

    if variant.upset_conflict:
        targets = conflict_upset_outcomes(row, legacy)
        new_upset = low_event_score(cells, {selected_bucket, second_bucket}, targets, excluded)
        if new_upset is not None:
            updated["upset_score"] = score_text((new_upset[0], new_upset[1]))
            updated["adjusted_score_4_upset"] = score_text((new_upset[0], new_upset[1]))
            updated["upset_score_probability"] = f"{new_upset[2]:.6f}"
            updated["adjusted_score_4_upset_probability"] = f"{new_upset[2]:.6f}"
            changed.append(f"upset_{'/'.join(sorted(targets))}")

    updated["conflict_score_variant"] = variant.name
    updated["conflict_score_notes"] = (
        f"legacy={legacy}; predicted={predicted}; second={second_bucket}; " + ",".join(changed or ["unchanged"])
    )
    return updated


def variants() -> list[Variant]:
    return [
        Variant("backup_legacy", False, True, False, True),
        Variant("upset_conflict", False, False, True, True),
        Variant("backup_plus_upset", False, True, True, True),
        Variant("model_conservative", True, False, False, True),
        Variant("model_backup", True, True, False, True),
        Variant("model_backup_upset", True, True, True, True),
        Variant("backup_no_same_draw", False, True, True, False),
    ]


def metric_delta(baseline: dict, current: dict) -> tuple[float, int, int, int, int]:
    return (
        baseline["mean_deviation"] - current["mean_deviation"],
        current["any_exact"] - baseline["any_exact"],
        current["any_score_bucket"] - baseline["any_score_bucket"],
        current["any_score_outcome"] - baseline["any_score_outcome"],
        current["top2_hit"] - baseline["top2_hit"],
    )


def format_metrics(metrics: dict) -> str:
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


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    rows = read_csv(PLAN_CSV)
    baseline = evaluate(PLAN_CSV)
    fieldnames = list(rows[0])
    for field in ("conflict_score_variant", "conflict_score_notes"):
        if field not in fieldnames:
            fieldnames.append(field)

    outputs: list[tuple[Variant, Path, dict]] = []
    for variant in variants():
        output_path = OUTPUT_DIR / f"outcome_conflict_score_{variant.name}.csv"
        variant_rows = [apply_variant(row, variant) for row in rows]
        write_csv(output_path, variant_rows, fieldnames)
        outputs.append((variant, output_path, evaluate(output_path)))

    outputs.sort(key=lambda item: metric_delta(baseline, item[2]), reverse=True)
    best_variant, best_path, best_metrics = outputs[0]

    before_rows = read_csv(PLAN_CSV)
    after_rows = read_csv(best_path)
    detail_rows = []
    for before, after in zip(before_rows, after_rows, strict=True):
        if before.get("outcome_edge_conflict") != "TRUE":
            continue
        if any(before[column] != after[column] for column in SCORE_COLUMNS):
            detail_rows.append(
                {
                    "date_bjt": after["date_bjt"],
                    "time_bjt": after["time_bjt"],
                    "match": f"{after['team_a']} vs {after['team_b']}",
                    "bucket": after["adjusted_total_goal_bucket"],
                    "old_scores": " / ".join(before[column] for column in SCORE_COLUMNS),
                    "new_scores": " / ".join(after[column] for column in SCORE_COLUMNS),
                    "xg_edge": after["xg_outcome_edge"],
                    "legacy_edge": after["legacy_outcome_edge"],
                    "notes": after["conflict_score_notes"],
                }
            )
    if detail_rows:
        write_csv(DETAIL_CSV, detail_rows, list(detail_rows[0]))

    lines = [
        "# Outcome Conflict Score Experiment",
        "",
        "- 只处理 `outcome_edge_conflict=TRUE` 的比赛。",
        "- 总比分桶不变，胜负参考不变。",
        "",
        "| Variant | Outcome | Top1 | Top2 | Exact | Score bucket | Score outcome | Mean dev | Median dev |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| current | {format_metrics(baseline)} |",
    ]
    for variant, _, metrics in outputs:
        lines.append(f"| {variant.name} | {format_metrics(metrics)} |")
    lines.extend(
        [
            "",
            f"Best variant: `{best_variant.name}`",
            f"Best output: `{best_path}`",
            f"Changed rows detail: `{DETAIL_CSV}`",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary: {SUMMARY_MD}")
    print(f"Best: {best_variant.name}")
    print(f"Baseline: {baseline}")
    print(f"Best: {best_metrics}")


if __name__ == "__main__":
    main()
