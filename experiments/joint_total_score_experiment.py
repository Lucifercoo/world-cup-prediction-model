from __future__ import annotations

import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import predict_fifa_profile as profile_model
from evaluation.evaluate_plan_against_results import evaluate


PLAN_CSV = ROOT / "output" / "realtime_context_adjusted_plan.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SUMMARY_MD = OUTPUT_DIR / "joint_total_score_experiment_summary.md"
DETAIL_CSV = OUTPUT_DIR / "joint_total_score_experiment_details.csv"
BUCKET_PROB_RE = re.compile(r"(\S+球)\s+([0-9.]+)%")
SCORE_COLUMNS = [
    "adjusted_score_1_model",
    "adjusted_score_2_aggressive_prediction",
    "adjusted_score_3_market_value",
    "adjusted_score_4_upset",
]


@dataclass(frozen=True)
class Variant:
    name: str
    bucket_weight: float
    score_weight: float
    outcome_weight: float
    btts_weight: float
    keep_bucket_output: bool
    selected_bucket_bonus: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_text(score: tuple[int, int]) -> str:
    return f"{score[0]}-{score[1]}"


def score_outcome(score: tuple[int, int]) -> str:
    if score[0] > score[1]:
        return "A"
    if score[1] > score[0]:
        return "B"
    return "D"


def score_bucket(score: tuple[int, int]) -> str:
    return profile_model.total_goal_bucket(score[0] + score[1])


def parse_bucket_probabilities(row: dict[str, str]) -> dict[str, float]:
    probabilities = {bucket: 0.0 for bucket in profile_model.TOTAL_GOAL_BUCKET_LABELS}
    for bucket, value in BUCKET_PROB_RE.findall(row["adjusted_total_goals_top2"]):
        probabilities[bucket] = float(value) / 100.0
    total = sum(probabilities.values())
    if total <= 0:
        probabilities[row["adjusted_total_goal_bucket"]] = 1.0
        total = 1.0
    return {bucket: value / total for bucket, value in probabilities.items()}


def format_bucket_probabilities(probabilities: dict[str, float], selected_bucket: str, second_bucket: str) -> str:
    priority = {selected_bucket: 0, second_bucket: 1}
    ordered = sorted(
        probabilities.items(),
        key=lambda item: (priority.get(item[0], 2), -item[1]),
    )
    return "; ".join(f"{bucket} {probability:.1%}" for bucket, probability in ordered)


def xg_cells(lambda_a: float, lambda_b: float) -> list[tuple[int, int, float]]:
    matrix = profile_model.score_matrix(lambda_a, lambda_b)
    cells = [
        (goals_a, goals_b, probability)
        for goals_a, row in enumerate(matrix)
        for goals_b, probability in enumerate(row)
    ]
    return sorted(cells, key=lambda cell: cell[2], reverse=True)


def outcome_probabilities(cells: list[tuple[int, int, float]]) -> dict[str, float]:
    probabilities = {"A": 0.0, "D": 0.0, "B": 0.0}
    for goals_a, goals_b, probability in cells:
        probabilities[score_outcome((goals_a, goals_b))] += probability
    total = sum(probabilities.values())
    if total <= 0:
        raise RuntimeError("xG outcome probabilities are empty")
    return {key: value / total for key, value in probabilities.items()}


def btts_probability(lambda_a: float, lambda_b: float) -> float:
    return (1.0 - math.exp(-lambda_a)) * (1.0 - math.exp(-lambda_b))


def raw_score_value(
    cell: tuple[int, int, float],
    bucket_probabilities: dict[str, float],
    outcome_probabilities_map: dict[str, float],
    predicted_outcome: str,
    selected_bucket: str,
    btts: float,
    variant: Variant,
) -> float:
    goals_a, goals_b, score_probability = cell
    score = (goals_a, goals_b)
    bucket = score_bucket(score)
    outcome = score_outcome(score)
    bucket_probability = max(bucket_probabilities.get(bucket, 0.0), 0.0001)
    outcome_probability = max(outcome_probabilities_map[outcome], 0.0001)
    value = (bucket_probability**variant.bucket_weight) * (score_probability**variant.score_weight)
    value *= outcome_probability**variant.outcome_weight
    if outcome == predicted_outcome:
        value *= 1.10
    elif outcome == "D" and abs(outcome_probabilities_map[predicted_outcome] - outcome_probability) <= 0.12:
        value *= 1.04
    if goals_a > 0 and goals_b > 0:
        value *= 1.0 + variant.btts_weight * btts
    if bucket == selected_bucket:
        value *= 1.0 + variant.selected_bucket_bonus
    return value


def choose_scores(row: dict[str, str], variant: Variant) -> tuple[list[tuple[int, int, float]], dict[str, float], dict[str, float], float]:
    lambda_a = float(row["adjusted_xg_a"])
    lambda_b = float(row["adjusted_xg_b"])
    cells = xg_cells(lambda_a, lambda_b)
    bucket_probabilities = parse_bucket_probabilities(row)
    outcome_probabilities_map = outcome_probabilities(cells)
    predicted_outcome = row["predicted_outcome"]
    selected_bucket = row["adjusted_total_goal_bucket"]
    btts = btts_probability(lambda_a, lambda_b)
    ranked = sorted(
        cells,
        key=lambda cell: raw_score_value(
            cell,
            bucket_probabilities,
            outcome_probabilities_map,
            predicted_outcome,
            selected_bucket,
            btts,
            variant,
        ),
        reverse=True,
    )
    return ranked, bucket_probabilities, outcome_probabilities_map, btts


def pick_distinct_scores(
    ranked: list[tuple[int, int, float]],
    bucket_order: list[str],
    required_outcomes: list[set[str] | None],
) -> list[tuple[int, int, float]]:
    selected: list[tuple[int, int, float]] = []
    used: set[tuple[int, int]] = set()
    for index, bucket in enumerate(bucket_order):
        allowed_outcomes = required_outcomes[index] if index < len(required_outcomes) else None
        candidates = [
            cell
            for cell in ranked
            if (cell[0], cell[1]) not in used
            and score_bucket((cell[0], cell[1])) == bucket
            and (allowed_outcomes is None or score_outcome((cell[0], cell[1])) in allowed_outcomes)
        ]
        if not candidates:
            candidates = [
                cell
                for cell in ranked
                if (cell[0], cell[1]) not in used and score_bucket((cell[0], cell[1])) == bucket
            ]
        if not candidates:
            candidates = [cell for cell in ranked if (cell[0], cell[1]) not in used]
        chosen = candidates[0]
        selected.append(chosen)
        used.add((chosen[0], chosen[1]))
    return selected


def apply_variant(row: dict[str, str], variant: Variant) -> dict[str, str]:
    updated = dict(row)
    ranked, bucket_probabilities, xg_outcomes, btts = choose_scores(row, variant)
    selected_bucket = row["adjusted_total_goal_bucket"]
    if variant.keep_bucket_output:
        model_bucket = selected_bucket
    else:
        model_bucket = score_bucket((ranked[0][0], ranked[0][1]))
    bucket_order = sorted(
        bucket_probabilities,
        key=lambda bucket: (bucket != model_bucket, -bucket_probabilities[bucket]),
    )
    if bucket_order[0] != model_bucket:
        bucket_order.remove(model_bucket)
        bucket_order.insert(0, model_bucket)
    second_bucket = next(bucket for bucket in bucket_order if bucket != model_bucket)
    predicted_outcome = row["predicted_outcome"]
    upset_outcomes = {"A", "D", "B"} - {predicted_outcome}
    if predicted_outcome != "D":
        upset_outcomes.add("D")

    selected = pick_distinct_scores(
        ranked,
        [model_bucket, second_bucket, model_bucket, second_bucket],
        [{predicted_outcome, "D"}, None, None, upset_outcomes],
    )
    model, backup, _market_placeholder, upset = selected

    updated["adjusted_total_goal_bucket"] = model_bucket
    updated["adjusted_total_goals_top2"] = format_bucket_probabilities(bucket_probabilities, model_bucket, second_bucket)
    updated["bucket_primary_score"] = score_text((model[0], model[1]))
    updated["adjusted_score_1_model"] = score_text((model[0], model[1]))
    updated["bucket_complement_score"] = score_text((backup[0], backup[1]))
    updated["aggressive_score"] = score_text((backup[0], backup[1]))
    updated["adjusted_score_2_aggressive_prediction"] = score_text((backup[0], backup[1]))
    updated["upset_score"] = score_text((upset[0], upset[1]))
    updated["upset_score_probability"] = f"{upset[2]:.6f}"
    updated["adjusted_score_4_upset"] = score_text((upset[0], upset[1]))
    updated["adjusted_score_4_upset_probability"] = f"{upset[2]:.6f}"
    updated["joint_total_score_variant"] = variant.name
    updated["joint_total_score_notes"] = (
        f"bucket={model_bucket}; second={second_bucket}; "
        f"xg A {xg_outcomes['A']:.1%}/D {xg_outcomes['D']:.1%}/B {xg_outcomes['B']:.1%}; "
        f"BTTS {btts:.1%}"
    )
    return updated


def variants() -> list[Variant]:
    return [
        Variant("joint_keep_bucket", 1.00, 0.70, 0.25, 0.20, True, 0.10),
        Variant("joint_keep_bucket_btts", 1.00, 0.70, 0.25, 0.45, True, 0.10),
        Variant("joint_free_bucket", 1.00, 0.70, 0.25, 0.20, False, 0.00),
        Variant("joint_free_bucket_btts", 1.00, 0.70, 0.25, 0.45, False, 0.00),
        Variant("joint_bucket_heavy", 1.35, 0.55, 0.20, 0.25, False, 0.00),
        Variant("joint_score_heavy", 0.75, 0.90, 0.30, 0.35, False, 0.00),
        Variant("joint_keep_bucket_strong", 1.20, 0.55, 0.20, 0.35, True, 0.25),
    ]


def metric_delta(baseline: dict, current: dict) -> tuple[float, int, int, int, int]:
    return (
        baseline["mean_deviation"] - current["mean_deviation"],
        current["any_exact"] - baseline["any_exact"],
        current["any_score_bucket"] - baseline["any_score_bucket"],
        current["top1_hit"] - baseline["top1_hit"],
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
    for field in ("joint_total_score_variant", "joint_total_score_notes"):
        if field not in fieldnames:
            fieldnames.append(field)

    outputs: list[tuple[Variant, Path, dict]] = []
    for variant in variants():
        output_path = OUTPUT_DIR / f"joint_total_score_{variant.name}.csv"
        variant_rows = [apply_variant(row, variant) for row in rows]
        write_csv(output_path, variant_rows, fieldnames)
        outputs.append((variant, output_path, evaluate(output_path)))
    outputs.sort(key=lambda item: metric_delta(baseline, item[2]), reverse=True)
    best_variant, best_path, best_metrics = outputs[0]

    before_rows = read_csv(PLAN_CSV)
    after_rows = read_csv(best_path)
    detail_rows = []
    for before, after in zip(before_rows, after_rows, strict=True):
        changed = (
            before["adjusted_total_goal_bucket"] != after["adjusted_total_goal_bucket"]
            or any(before[column] != after[column] for column in SCORE_COLUMNS)
        )
        if changed:
            detail_rows.append(
                {
                    "date_bjt": after["date_bjt"],
                    "time_bjt": after["time_bjt"],
                    "match": f"{after['team_a']} vs {after['team_b']}",
                    "old_bucket": before["adjusted_total_goal_bucket"],
                    "new_bucket": after["adjusted_total_goal_bucket"],
                    "old_scores": " / ".join(before[column] for column in SCORE_COLUMNS),
                    "new_scores": " / ".join(after[column] for column in SCORE_COLUMNS),
                    "notes": after["joint_total_score_notes"],
                }
            )
    if detail_rows:
        write_csv(DETAIL_CSV, detail_rows, list(detail_rows[0]))

    lines = [
        "# Joint Total Score Experiment",
        "",
        "- 联合打分：每个比分用总进球桶概率、xG 比分概率、xG 胜平负概率和双方进球概率一起排序。",
        "- `keep_bucket` 版本不改总进球桶，只在桶内联合选比分；`free_bucket` 版本允许模型分反推总进球桶。",
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
