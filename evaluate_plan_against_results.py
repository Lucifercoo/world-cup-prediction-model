from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "data" / "world_cup_2026_results.csv"
OUTPUT_DIR = ROOT / "output"
DEFAULT_PLAN = OUTPUT_DIR / "realtime_context_adjusted_plan.csv"
BASELINE_PLAN = OUTPUT_DIR / "realtime_context_adjusted_plan_baseline_before_dual_bucket.csv"
SUMMARY_MD = OUTPUT_DIR / "dual_bucket_model_comparison.md"
SCORE_COLUMNS = [
    "adjusted_score_1_model",
    "adjusted_score_2_aggressive_prediction",
    "adjusted_score_3_market_value",
    "adjusted_score_4_upset",
]


def parse_score(value: str) -> tuple[int, int]:
    home, away = value.split("-", maxsplit=1)
    return int(home), int(away)


def format_score(home: int, away: int) -> str:
    return f"{home}-{away}"


def outcome(score: str) -> str:
    home, away = parse_score(score)
    if home > away:
        return "A"
    if away > home:
        return "B"
    return "D"


def total_bucket(total: int) -> str:
    if total <= 1:
        return "0-1球"
    if total <= 3:
        return "2-3球"
    if total <= 5:
        return "4-5球"
    return "6-8球"


def row_key(row: dict) -> tuple[str, str, str, str]:
    return row["date_bjt"], row["time_bjt"], row["team_a"], row["team_b"]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def parse_top2_buckets(value: str, selected_bucket: str) -> set[str]:
    buckets = {selected_bucket}
    for part in value.split(";"):
        piece = part.strip().split()
        if not piece:
            continue
        buckets.add(piece[0])
        if len(buckets) >= 2:
            break
    return buckets


def deviation(actual: str, predicted: str) -> float:
    actual_home, actual_away = parse_score(actual)
    predicted_home, predicted_away = parse_score(predicted)
    return (abs(actual_home - predicted_home) + abs(actual_away - predicted_away)) / max(
        1,
        actual_home + actual_away,
    )


def evaluate(plan_path: Path) -> dict:
    results = read_csv(RESULTS_CSV)
    plan = {row_key(row): row for row in read_csv(plan_path)}
    rows: list[dict] = []
    for result in results:
        key = row_key(result)
        if key not in plan:
            continue
        prediction = plan[key]
        actual_score = format_score(int(result["goals_a"]), int(result["goals_b"]))
        actual_bucket = total_bucket(sum(parse_score(actual_score)))
        selected_bucket = prediction["adjusted_total_goal_bucket"]
        top2_buckets = parse_top2_buckets(prediction["adjusted_total_goals_top2"], selected_bucket)
        score_deviations: list[float] = []
        any_exact = False
        any_score_bucket = False
        any_score_outcome = False
        for column in SCORE_COLUMNS:
            score = prediction.get(column, "")
            if not score:
                continue
            score_deviations.append(deviation(actual_score, score))
            any_exact = any_exact or score == actual_score
            any_score_bucket = any_score_bucket or total_bucket(sum(parse_score(score))) == actual_bucket
            any_score_outcome = any_score_outcome or outcome(score) == outcome(actual_score)
        rows.append(
            {
                "actual_bucket": actual_bucket,
                "selected_bucket": selected_bucket,
                "outcome_hit": prediction["predicted_outcome"] == outcome(actual_score),
                "top1_hit": selected_bucket == actual_bucket,
                "top2_hit": actual_bucket in top2_buckets,
                "any_exact": any_exact,
                "any_score_bucket": any_score_bucket,
                "any_score_outcome": any_score_outcome,
                "mean_deviation": sum(score_deviations) / len(score_deviations),
            }
        )
    n = len(rows)
    selected_2_3 = sum(row["selected_bucket"] == "2-3球" for row in rows)
    actual_2_3 = sum(row["actual_bucket"] == "2-3球" for row in rows)
    return {
        "matches": n,
        "outcome_hit": sum(row["outcome_hit"] for row in rows),
        "top1_hit": sum(row["top1_hit"] for row in rows),
        "top2_hit": sum(row["top2_hit"] for row in rows),
        "any_exact": sum(row["any_exact"] for row in rows),
        "any_score_bucket": sum(row["any_score_bucket"] for row in rows),
        "any_score_outcome": sum(row["any_score_outcome"] for row in rows),
        "mean_deviation": sum(row["mean_deviation"] for row in rows) / n if n else 0.0,
        "median_deviation": statistics.median(row["mean_deviation"] for row in rows) if rows else 0.0,
        "selected_2_3": selected_2_3,
        "actual_2_3": actual_2_3,
    }


def pct(value: float) -> str:
    return f"{value:.1%}"


def write_summary(baseline: dict, current: dict) -> None:
    def hit(summary: dict, key: str) -> str:
        n = max(1, summary["matches"])
        return f"{summary[key]}/{summary['matches']} {pct(summary[key] / n)}"

    lines = [
        "# Dual Bucket Model Comparison",
        "",
        "| Metric | Baseline | Current |",
        "|---|---:|---:|",
        f"| Matches | {baseline['matches']} | {current['matches']} |",
        f"| Outcome hit | {hit(baseline, 'outcome_hit')} | {hit(current, 'outcome_hit')} |",
        f"| Top1 bucket hit | {hit(baseline, 'top1_hit')} | {hit(current, 'top1_hit')} |",
        f"| Top2 bucket hit | {hit(baseline, 'top2_hit')} | {hit(current, 'top2_hit')} |",
        f"| Any exact score | {hit(baseline, 'any_exact')} | {hit(current, 'any_exact')} |",
        f"| Any score bucket | {hit(baseline, 'any_score_bucket')} | {hit(current, 'any_score_bucket')} |",
        f"| Mean deviation | {baseline['mean_deviation']:.3f} | {current['mean_deviation']:.3f} |",
        f"| Median deviation | {baseline['median_deviation']:.3f} | {current['median_deviation']:.3f} |",
        f"| Selected 2-3 | {hit(baseline, 'selected_2_3')} | {hit(current, 'selected_2_3')} |",
        f"| Actual 2-3 | {hit(baseline, 'actual_2_3')} | {hit(current, 'actual_2_3')} |",
    ]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    baseline_path = Path(sys.argv[1]) if len(sys.argv) > 1 else BASELINE_PLAN
    current_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PLAN
    baseline = evaluate(baseline_path)
    current = evaluate(current_path)
    write_summary(baseline, current)
    print(f"Summary: {SUMMARY_MD}")
    print(f"Baseline: {baseline}")
    print(f"Current: {current}")


if __name__ == "__main__":
    main()
