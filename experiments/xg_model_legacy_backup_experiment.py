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
SUMMARY_MD = OUTPUT_DIR / "xg_model_legacy_backup_experiment_summary.md"
DETAIL_CSV = OUTPUT_DIR / "xg_model_legacy_backup_experiment_details.csv"
SCORE_COLUMNS = [
    "adjusted_score_1_model",
    "adjusted_score_2_aggressive_prediction",
    "adjusted_score_3_market_value",
    "adjusted_score_4_upset",
]


@dataclass(frozen=True)
class Variant:
    name: str
    model_cells: str
    backup_cells: str
    upset_cells: str
    backup_bucket: str
    upset_mode: str
    upset_bucket: str = "top1_top2"


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


def opposite_outcome(outcome: str) -> str:
    if outcome == "A":
        return "B"
    if outcome == "B":
        return "A"
    raise ValueError("draw has no opposite outcome")


def raw_xg_cells(lambda_a: float, lambda_b: float) -> list[tuple[int, int, float]]:
    matrix = profile_model.score_matrix(lambda_a, lambda_b)
    cells = [
        (goals_a, goals_b, probability)
        for goals_a, row in enumerate(matrix)
        for goals_b, probability in enumerate(row)
    ]
    return sorted(cells, key=lambda cell: cell[2], reverse=True)


def legacy_cells(row: dict[str, str]) -> list[tuple[int, int, float]]:
    return profile_model.outcome_adjusted_scores(
        float(row["adjusted_xg_a"]),
        float(row["adjusted_xg_b"]),
        float(row["adjusted_p_a"]),
        float(row["adjusted_p_draw"]),
        float(row["adjusted_p_b"]),
    )


def cells_for(row: dict[str, str], kind: str) -> list[tuple[int, int, float]]:
    if kind == "xg":
        return raw_xg_cells(float(row["adjusted_xg_a"]), float(row["adjusted_xg_b"]))
    if kind == "legacy":
        return legacy_cells(row)
    raise ValueError(f"unknown cell kind: {kind}")


def ordered_buckets(row: dict[str, str]) -> list[str]:
    buckets = [row["adjusted_total_goal_bucket"]]
    for part in row["adjusted_total_goals_top2"].split(";"):
        piece = part.strip().split()
        if piece and piece[0] not in buckets:
            buckets.append(piece[0])
    for bucket in profile_model.TOTAL_GOAL_BUCKET_LABELS:
        if bucket not in buckets:
            buckets.append(bucket)
    return buckets


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
        and profile_model.total_goal_bucket(cell[0] + cell[1]) in buckets
        and score_outcome((cell[0], cell[1])) in outcomes
    ]


def best_score(
    cells: list[tuple[int, int, float]],
    buckets: set[str],
    outcomes: set[str],
    excluded: set[tuple[int, int]],
) -> tuple[int, int, float]:
    candidates = candidate_scores(cells, buckets, outcomes, excluded)
    if candidates:
        return candidates[0]
    relaxed = [
        cell
        for cell in cells
        if (cell[0], cell[1]) not in excluded and profile_model.total_goal_bucket(cell[0] + cell[1]) in buckets
    ]
    if relaxed:
        return relaxed[0]
    return next(cell for cell in cells if (cell[0], cell[1]) not in excluded)


def limited_xg_model_score(
    row: dict[str, str],
    cells: list[tuple[int, int, float]],
    bucket: str,
    predicted: str,
    excluded: set[tuple[int, int]],
) -> tuple[int, int, float]:
    base = best_score(cells, {bucket}, {predicted}, excluded)
    goals_a, goals_b, _ = base
    total = goals_a + goals_b
    if total < 4 or min(goals_a, goals_b) > 0 or abs(goals_a - goals_b) < 3:
        return base

    lambda_a = float(row["adjusted_xg_a"])
    lambda_b = float(row["adjusted_xg_b"])
    p_draw = float(row["adjusted_p_draw"])
    loser_lambda = lambda_b if goals_a > goals_b else lambda_a
    if loser_lambda < 0.42 and p_draw < 0.24:
        return base

    guarded = [
        cell
        for cell in cells
        if (cell[0], cell[1]) not in excluded
        and cell[0] + cell[1] == total
        and score_outcome((cell[0], cell[1])) == score_outcome((goals_a, goals_b))
        and min(cell[0], cell[1]) >= 1
    ]
    return guarded[0] if guarded else base


def upset_outcomes(row: dict[str, str], mode: str, model_outcome: str) -> set[str]:
    predicted = row["predicted_outcome"]
    probabilities = {
        "A": float(row["adjusted_p_a"]),
        "D": float(row["adjusted_p_draw"]),
        "B": float(row["adjusted_p_b"]),
    }
    favorite = max(probabilities.items(), key=lambda item: item[1])[0]
    favorite_probability = probabilities[favorite]
    if mode == "xg_inverse":
        if predicted == "D":
            return {"A", "B"}
        return {"D", opposite_outcome(predicted)}
    if mode == "different_from_model":
        if model_outcome == "D":
            return {"A", "B"}
        return {"D", opposite_outcome(model_outcome)}
    if mode == "legacy_upset":
        if favorite == "D":
            return {"A", "B"}
        underdog = "B" if favorite == "A" else "A"
        if favorite_probability >= 0.70:
            return {favorite}
        return {"D", underdog}
    raise ValueError(f"unknown upset mode: {mode}")


def apply_variant(row: dict[str, str], variant: Variant) -> dict[str, str]:
    updated = dict(row)
    bucket_order = ordered_buckets(row)
    selected_bucket = bucket_order[0]
    second_bucket = bucket_order[1]
    predicted = row["predicted_outcome"]

    if variant.model_cells == "xg_limited":
        model_cells = cells_for(row, "xg")
        model = limited_xg_model_score(row, model_cells, selected_bucket, predicted, set())
    else:
        model_cells = cells_for(row, variant.model_cells)
        model = best_score(model_cells, {selected_bucket}, {predicted}, set())
    excluded = {(model[0], model[1])}
    model_outcome = score_outcome((model[0], model[1]))

    backup_cells = cells_for(row, variant.backup_cells)
    backup_buckets = {second_bucket} if variant.backup_bucket == "top2" else {selected_bucket, second_bucket}
    backup = best_score(backup_cells, backup_buckets, {predicted}, excluded)
    excluded.add((backup[0], backup[1]))

    market = parse_score(row["adjusted_score_3_market_value"])
    excluded.add(market)

    upset_cells = cells_for(row, variant.upset_cells)
    upset_buckets = {selected_bucket, second_bucket}
    if variant.upset_bucket == "top2_top3":
        upset_buckets = set(bucket_order[1:3])
    upset = best_score(
        upset_cells,
        upset_buckets,
        upset_outcomes(row, variant.upset_mode, model_outcome),
        excluded,
    )

    updated["adjusted_score_1_model"] = score_text((model[0], model[1]))
    updated["bucket_primary_score"] = updated["adjusted_score_1_model"]
    updated["adjusted_score_2_aggressive_prediction"] = score_text((backup[0], backup[1]))
    updated["aggressive_score"] = updated["adjusted_score_2_aggressive_prediction"]
    updated["bucket_complement_score"] = updated["adjusted_score_2_aggressive_prediction"]
    updated["adjusted_score_4_upset"] = score_text((upset[0], upset[1]))
    updated["upset_score"] = updated["adjusted_score_4_upset"]
    updated["upset_score_probability"] = f"{upset[2]:.6f}"
    updated["adjusted_score_4_upset_probability"] = f"{upset[2]:.6f}"
    updated["xg_model_legacy_backup_variant"] = variant.name
    updated["xg_model_legacy_backup_notes"] = (
        f"model={variant.model_cells}:{selected_bucket}; "
        f"backup={variant.backup_cells}:{variant.backup_bucket}; "
        f"upset={variant.upset_cells}:{variant.upset_mode}:{variant.upset_bucket}"
    )
    return updated


def variants() -> list[Variant]:
    return [
        Variant("xg_model_legacy_top2_xg_inverse", "xg", "legacy", "xg", "top2", "xg_inverse"),
        Variant("xg_model_legacy_top2_legacy_upset", "xg", "legacy", "legacy", "top2", "legacy_upset"),
        Variant("xg_model_legacy_any_xg_inverse", "xg", "legacy", "xg", "any", "xg_inverse"),
        Variant("xg_model_xg_top2_xg_inverse", "xg", "xg", "xg", "top2", "xg_inverse"),
        Variant("legacy_model_xg_top2_xg_inverse", "legacy", "xg", "xg", "top2", "xg_inverse"),
        Variant(
            "xg_limited_model_legacy_any_upset_top23_xg",
            "xg_limited",
            "legacy",
            "xg",
            "any",
            "different_from_model",
            "top2_top3",
        ),
        Variant(
            "xg_limited_model_legacy_any_upset_top23_legacy",
            "xg_limited",
            "legacy",
            "legacy",
            "any",
            "different_from_model",
            "top2_top3",
        ),
        Variant(
            "xg_limited_model_legacy_top2_upset_top23_legacy",
            "xg_limited",
            "legacy",
            "legacy",
            "top2",
            "different_from_model",
            "top2_top3",
        ),
    ]


def metric_delta(baseline: dict, current: dict) -> tuple[float, int, int, int, int, float]:
    return (
        baseline["mean_deviation"] - current["mean_deviation"],
        current["any_exact"] - baseline["any_exact"],
        current["any_score_bucket"] - baseline["any_score_bucket"],
        current["any_score_outcome"] - baseline["any_score_outcome"],
        current["top2_hit"] - baseline["top2_hit"],
        baseline["median_deviation"] - current["median_deviation"],
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
    for field in ("xg_model_legacy_backup_variant", "xg_model_legacy_backup_notes"):
        if field not in fieldnames:
            fieldnames.append(field)

    outputs: list[tuple[Variant, Path, dict]] = []
    for variant in variants():
        output_path = OUTPUT_DIR / f"xg_model_legacy_backup_{variant.name}.csv"
        variant_rows = [apply_variant(row, variant) for row in rows]
        write_csv(output_path, variant_rows, fieldnames)
        outputs.append((variant, output_path, evaluate(output_path)))

    outputs.sort(key=lambda item: metric_delta(baseline, item[2]), reverse=True)
    best_variant, best_path, best_metrics = outputs[0]

    before_rows = read_csv(PLAN_CSV)
    after_rows = read_csv(best_path)
    detail_rows = []
    for before, after in zip(before_rows, after_rows, strict=True):
        if any(before[column] != after[column] for column in SCORE_COLUMNS):
            detail_rows.append(
                {
                    "date_bjt": after["date_bjt"],
                    "time_bjt": after["time_bjt"],
                    "match": f"{after['team_a']} vs {after['team_b']}",
                    "bucket": after["adjusted_total_goal_bucket"],
                    "top2": after["adjusted_total_goals_top2"],
                    "old_scores": " / ".join(before[column] for column in SCORE_COLUMNS),
                    "new_scores": " / ".join(after[column] for column in SCORE_COLUMNS),
                    "notes": after["xg_model_legacy_backup_notes"],
                }
            )
    if detail_rows:
        write_csv(DETAIL_CSV, detail_rows, list(detail_rows[0]))

    lines = [
        "# xG Model + Legacy Backup Experiment",
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
