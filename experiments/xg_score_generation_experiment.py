from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import predict_fifa_profile as profile_model
from evaluation.evaluate_plan_against_results import evaluate


PLAN_CSV = ROOT / "output" / "realtime_context_adjusted_plan.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SUMMARY_MD = OUTPUT_DIR / "xg_score_generation_experiment_summary.md"
DETAIL_CSV = OUTPUT_DIR / "xg_score_generation_experiment_details.csv"


SCORE_COLUMNS = [
    "adjusted_score_1_model",
    "adjusted_score_2_aggressive_prediction",
    "adjusted_score_3_market_value",
    "adjusted_score_4_upset",
]


@dataclass(frozen=True)
class Variant:
    name: str
    model_policy: str
    backup_policy: str
    btts_bonus: float
    draw_allowance: float
    upset_policy: str


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


def outcome(score: tuple[int, int]) -> str:
    if score[0] > score[1]:
        return "A"
    if score[1] > score[0]:
        return "B"
    return "D"


def parse_bucket_list(value: str, selected_bucket: str) -> list[str]:
    buckets = [selected_bucket]
    for part in value.split(";"):
        piece = part.strip().split()
        if piece and piece[0] not in buckets:
            buckets.append(piece[0])
    for bucket in profile_model.TOTAL_GOAL_BUCKET_LABELS:
        if bucket not in buckets:
            buckets.append(bucket)
    return buckets


def xg_cells(lambda_a: float, lambda_b: float) -> list[tuple[int, int, float]]:
    matrix = profile_model.score_matrix(lambda_a, lambda_b)
    cells = [
        (goals_a, goals_b, probability)
        for goals_a, row in enumerate(matrix)
        for goals_b, probability in enumerate(row)
    ]
    return sorted(cells, key=lambda cell: cell[2], reverse=True)


def xg_outcome_probabilities(cells: list[tuple[int, int, float]]) -> dict[str, float]:
    probabilities = {"A": 0.0, "D": 0.0, "B": 0.0}
    for goals_a, goals_b, probability in cells:
        probabilities[outcome((goals_a, goals_b))] += probability
    total = sum(probabilities.values())
    if total <= 0:
        raise RuntimeError("empty xG outcome probabilities")
    return {key: value / total for key, value in probabilities.items()}


def bucket_of_score(score: tuple[int, int]) -> str:
    return profile_model.total_goal_bucket(score[0] + score[1])


def btts_probability(lambda_a: float, lambda_b: float) -> float:
    return (1.0 - math.exp(-lambda_a)) * (1.0 - math.exp(-lambda_b))


def allowed_outcomes(
    policy: str,
    predicted_outcome: str,
    xg_outcomes: dict[str, float],
    draw_allowance: float,
) -> set[str]:
    if policy == "free":
        return {"A", "D", "B"}
    if policy == "strict":
        return {predicted_outcome}
    if policy == "draw_soft":
        if predicted_outcome == "D":
            return {"D"}
        if max(xg_outcomes["A"], xg_outcomes["B"]) - xg_outcomes["D"] <= draw_allowance:
            return {predicted_outcome, "D"}
        return {predicted_outcome}
    if policy == "non_main":
        return {"A", "D", "B"} - {predicted_outcome}
    if policy == "underdog_or_draw":
        if predicted_outcome == "A":
            return {"D", "B"}
        if predicted_outcome == "B":
            return {"A", "D"}
        return {"A", "B"}
    raise ValueError(f"unknown outcome policy: {policy}")


def select_score(
    cells: list[tuple[int, int, float]],
    buckets: set[str],
    allowed: set[str],
    xg_outcomes: dict[str, float],
    btts_bonus: float,
    btts: float,
    excluded_scores: set[tuple[int, int]] | None = None,
) -> tuple[int, int, float]:
    excluded_scores = excluded_scores or set()
    candidates = [
        cell
        for cell in cells
        if (cell[0], cell[1]) not in excluded_scores
        and bucket_of_score((cell[0], cell[1])) in buckets
        and outcome((cell[0], cell[1])) in allowed
    ]
    if not candidates:
        candidates = [
            cell
            for cell in cells
            if (cell[0], cell[1]) not in excluded_scores and bucket_of_score((cell[0], cell[1])) in buckets
        ]
    if not candidates:
        raise RuntimeError(f"no candidate score in buckets={sorted(buckets)}")

    def rank(cell: tuple[int, int, float]) -> tuple[float, float, float]:
        goals_a, goals_b, probability = cell
        score_outcome = outcome((goals_a, goals_b))
        value = probability
        value *= 0.75 + xg_outcomes[score_outcome]
        if goals_a > 0 and goals_b > 0:
            value *= 1.0 + btts_bonus * btts
        return value, probability, -(goals_a + goals_b)

    return max(candidates, key=rank)


def select_upset_score(
    cells: list[tuple[int, int, float]],
    selected_bucket: str,
    second_bucket: str,
    predicted_outcome: str,
    xg_outcomes: dict[str, float],
    variant: Variant,
    btts: float,
    excluded_scores: set[tuple[int, int]],
) -> tuple[int, int, float]:
    if variant.upset_policy == "low_non_main":
        preferred_buckets = {selected_bucket}
        if selected_bucket in {"4-5球", "6-8球"}:
            preferred_buckets = {"0-1球", "2-3球"}
        allowed = allowed_outcomes("underdog_or_draw", predicted_outcome, xg_outcomes, variant.draw_allowance)
        return select_score(cells, preferred_buckets, allowed, xg_outcomes, 0.0, btts, excluded_scores)

    allowed = allowed_outcomes("non_main", predicted_outcome, xg_outcomes, variant.draw_allowance)
    return select_score(cells, {selected_bucket, second_bucket}, allowed, xg_outcomes, 0.0, btts, excluded_scores)


def apply_variant(row: dict[str, str], variant: Variant) -> dict[str, str]:
    updated = dict(row)
    lambda_a = float(row["adjusted_xg_a"])
    lambda_b = float(row["adjusted_xg_b"])
    cells = xg_cells(lambda_a, lambda_b)
    xg_outcomes = xg_outcome_probabilities(cells)
    selected_bucket = row["adjusted_total_goal_bucket"]
    bucket_list = parse_bucket_list(row["adjusted_total_goals_top2"], selected_bucket)
    second_bucket = next(bucket for bucket in bucket_list if bucket != selected_bucket)
    predicted_outcome = row["predicted_outcome"]
    btts = btts_probability(lambda_a, lambda_b)

    model_allowed = allowed_outcomes(
        variant.model_policy,
        predicted_outcome,
        xg_outcomes,
        variant.draw_allowance,
    )
    model = select_score(
        cells,
        {selected_bucket},
        model_allowed,
        xg_outcomes,
        variant.btts_bonus,
        btts,
    )
    excluded = {(model[0], model[1])}
    backup_allowed = allowed_outcomes(
        variant.backup_policy,
        predicted_outcome,
        xg_outcomes,
        variant.draw_allowance,
    )
    backup = select_score(
        cells,
        {second_bucket},
        backup_allowed,
        xg_outcomes,
        variant.btts_bonus,
        btts,
        excluded,
    )
    excluded.add((backup[0], backup[1]))
    upset = select_upset_score(
        cells,
        selected_bucket,
        second_bucket,
        predicted_outcome,
        xg_outcomes,
        variant,
        btts,
        excluded,
    )

    updated["bucket_primary_score"] = score_text((model[0], model[1]))
    updated["adjusted_score_1_model"] = score_text((model[0], model[1]))
    updated["bucket_complement_score"] = score_text((backup[0], backup[1]))
    updated["aggressive_score"] = score_text((backup[0], backup[1]))
    updated["adjusted_score_2_aggressive_prediction"] = score_text((backup[0], backup[1]))
    updated["upset_score"] = score_text((upset[0], upset[1]))
    updated["upset_score_probability"] = f"{upset[2]:.6f}"
    updated["adjusted_score_4_upset"] = score_text((upset[0], upset[1]))
    updated["adjusted_score_4_upset_probability"] = f"{upset[2]:.6f}"
    updated["xg_score_variant"] = variant.name
    updated["xg_outcome_probabilities"] = (
        f"A {xg_outcomes['A']:.1%}; D {xg_outcomes['D']:.1%}; B {xg_outcomes['B']:.1%}; BTTS {btts:.1%}"
    )
    return updated


def variants() -> list[Variant]:
    return [
        Variant("strict_raw", "strict", "strict", 0.0, 0.08, "non_main"),
        Variant("free_raw", "free", "free", 0.0, 0.08, "non_main"),
        Variant("draw_soft", "draw_soft", "draw_soft", 0.0, 0.12, "non_main"),
        Variant("strict_btts", "strict", "strict", 0.45, 0.08, "non_main"),
        Variant("draw_soft_btts", "draw_soft", "draw_soft", 0.45, 0.12, "non_main"),
        Variant("free_btts", "free", "free", 0.45, 0.08, "non_main"),
        Variant("draw_soft_low_upset", "draw_soft", "draw_soft", 0.45, 0.12, "low_non_main"),
    ]


def metric_delta(baseline: dict, current: dict) -> tuple[float, int, int, int]:
    return (
        baseline["mean_deviation"] - current["mean_deviation"],
        current["any_exact"] - baseline["any_exact"],
        current["any_score_bucket"] - baseline["any_score_bucket"],
        current["any_score_outcome"] - baseline["any_score_outcome"],
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
    baseline_metrics = evaluate(PLAN_CSV)
    fieldnames = list(rows[0])
    for extra in ("xg_score_variant", "xg_outcome_probabilities"):
        if extra not in fieldnames:
            fieldnames.append(extra)

    outputs: list[tuple[Variant, Path, dict]] = []
    for variant in variants():
        output_path = OUTPUT_DIR / f"xg_score_generation_{variant.name}.csv"
        variant_rows = [apply_variant(row, variant) for row in rows]
        write_csv(output_path, variant_rows, fieldnames)
        outputs.append((variant, output_path, evaluate(output_path)))

    outputs.sort(key=lambda item: metric_delta(baseline_metrics, item[2]), reverse=True)
    best_variant, best_path, best_metrics = outputs[0]

    detail_rows: list[dict[str, str]] = []
    baseline_rows = read_csv(PLAN_CSV)
    best_rows = read_csv(best_path)
    for before, after in zip(baseline_rows, best_rows, strict=True):
        changed = any(before.get(column) != after.get(column) for column in SCORE_COLUMNS)
        if changed:
            detail_rows.append(
                {
                    "date_bjt": after["date_bjt"],
                    "time_bjt": after["time_bjt"],
                    "match": f"{after['team_a']} vs {after['team_b']}",
                    "bucket": after["adjusted_total_goal_bucket"],
                    "top2": after["adjusted_total_goals_top2"],
                    "predicted_outcome": after["predicted_outcome"],
                    "old_scores": " / ".join(before.get(column, "") for column in SCORE_COLUMNS),
                    "new_scores": " / ".join(after.get(column, "") for column in SCORE_COLUMNS),
                    "xg_outcomes": after["xg_outcome_probabilities"],
                }
            )
    if detail_rows:
        write_csv(DETAIL_CSV, detail_rows, list(detail_rows[0]))

    lines = [
        "# xG Score Generation Experiment",
        "",
        "- 只替换四个比分列；胜负、总进球桶、实时上下文不改。",
        "- `身价`列暂不改，避免把身价信号也混入本次实验。",
        "",
        "| Variant | Outcome | Top1 | Top2 | Exact | Score bucket | Score outcome | Mean dev | Median dev |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| current | {format_metrics(baseline_metrics)} |",
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
    print(f"Baseline: {baseline_metrics}")
    print(f"Best: {best_metrics}")


if __name__ == "__main__":
    main()
