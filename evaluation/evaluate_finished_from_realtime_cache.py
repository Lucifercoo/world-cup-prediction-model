from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
RESULTS_CSV = ROOT / "data" / "world_cup_2026_results.csv"
KNOCKOUT_DECISIONS_CSV = ROOT / "data" / "world_cup_2026_knockout_decisions.csv"
CACHE_DIR = OUTPUT_DIR / "realtime_context_cache"
DETAIL_CSV = OUTPUT_DIR / "finished_realtime_cache_evaluation.csv"
SUMMARY_MD = OUTPUT_DIR / "finished_realtime_cache_evaluation_summary.md"
BUCKET_REWEIGHT_MD = OUTPUT_DIR / "finished_realtime_cache_bucket_reweight_experiment.md"
TOTAL_GOAL_BUCKET_REWEIGHT_FACTORS = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
SCORE_COLUMNS = [
    ("model", "adjusted_score_1_model"),
    ("aggressive", "adjusted_score_2_aggressive_prediction"),
    ("market", "adjusted_score_3_market_value"),
    ("upset", "adjusted_score_4_upset"),
]
FALLBACK_SCORE_COLUMNS = {
    "adjusted_score_1_model": "bucket_primary_score",
    "adjusted_score_2_aggressive_prediction": "aggressive_score",
    "adjusted_score_3_market_value": "market_value_score",
    "adjusted_score_4_upset": "upset_score",
}


def parse_bjt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"cache timestamp lacks timezone: {value}")
    return parsed.astimezone(timezone.utc)


def parse_match_bjt(date_text: str, time_text: str) -> datetime:
    return datetime.strptime(f"{date_text} {time_text} +0800", "%Y-%m-%d %H:%M %z").astimezone(timezone.utc)


def parse_score(value: str) -> tuple[int, int]:
    home, away = value.split("-", maxsplit=1)
    return int(home), int(away)


def format_score(home: int, away: int) -> str:
    return f"{home}-{away}"


def outcome(score: tuple[int, int]) -> str:
    if score[0] > score[1]:
        return "A"
    if score[1] > score[0]:
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


def score_deviation(actual: str, predicted: str) -> float:
    actual_home, actual_away = parse_score(actual)
    predicted_home, predicted_away = parse_score(predicted)
    return (abs(predicted_home - actual_home) + abs(predicted_away - actual_away)) / max(
        1,
        actual_home + actual_away,
    )


def prediction_value(row: dict, column: str) -> str | None:
    value = row.get(column, "")
    if value:
        return value
    fallback = FALLBACK_SCORE_COLUMNS.get(column)
    if fallback is None:
        return None
    value = row.get(fallback, "")
    if not value:
        return None
    return value


def pct(value: float) -> str:
    return f"{value:.1%}"


def knockout_decision_key(row: dict) -> tuple[str, str, str, str]:
    return row["date_bjt"], row["time_bjt"], row["team_a"], row["team_b"]


def load_knockout_decisions() -> dict[tuple[str, str, str, str], dict]:
    decisions: dict[tuple[str, str, str, str], dict] = {}
    with KNOCKOUT_DECISIONS_CSV.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            decisions[knockout_decision_key(row)] = row
    return decisions


def load_results() -> list[dict]:
    rows: list[dict] = []
    decisions = load_knockout_decisions()
    with RESULTS_CSV.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            final_score = format_score(int(row["goals_a"]), int(row["goals_b"]))
            regulation_score = final_score
            evaluation_phase = "final"
            decision_method = ""
            advancing_team = ""
            decision = decisions.get(knockout_decision_key(row))
            if decision is not None:
                regulation_score = format_score(
                    int(decision["regulation_goals_a"]),
                    int(decision["regulation_goals_b"]),
                )
                final_score = format_score(
                    int(decision["final_goals_a"]),
                    int(decision["final_goals_b"]),
                )
                evaluation_phase = "regulation"
                decision_method = decision["decision_method"]
                advancing_team = decision["advancing_team"]
            rows.append(
                {
                    **row,
                    "actual_score": regulation_score,
                    "regulation_score": regulation_score,
                    "final_score": final_score,
                    "evaluation_phase": evaluation_phase,
                    "decision_method": decision_method,
                    "advancing_team": advancing_team,
                    "kickoff_utc": parse_match_bjt(row["date_bjt"], row["time_bjt"]),
                }
            )
    return rows


def load_cache_runs() -> list[dict]:
    runs: list[dict] = []
    if not CACHE_DIR.exists():
        return runs
    for manifest_path in CACHE_DIR.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_path = manifest_path.parent / "outputs" / "realtime_context_adjusted_plan.csv"
        if not output_path.exists():
            continue
        runs.append(
            {
                "run_id": manifest["run_id"],
                "created_at_utc": parse_bjt(manifest["created_at_bjt"]),
                "plan_csv": output_path,
            }
        )
    return sorted(runs, key=lambda item: item["created_at_utc"])


def row_key(row: dict) -> tuple[str, str, str, str]:
    return row["date_bjt"], row["time_bjt"], row["team_a"], row["team_b"]


def load_plan_rows(path: Path) -> dict[tuple[str, str, str, str], dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {row_key(row): row for row in csv.DictReader(fh)}


def cache_for_match(result: dict, runs: list[dict]) -> tuple[dict, dict] | None:
    candidates = [run for run in runs if run["created_at_utc"] < result["kickoff_utc"]]
    for run in reversed(candidates):
        rows = load_plan_rows(run["plan_csv"])
        key = (result["date_bjt"], result["time_bjt"], result["team_a"], result["team_b"])
        if key in rows:
            return run, rows[key]
    return None


def top2_buckets(value: str, primary: str) -> set[str]:
    buckets = {primary}
    for part in value.split(";"):
        bucket = part.strip().split(" ", maxsplit=1)[0]
        if bucket:
            buckets.add(bucket)
        if len(buckets) >= 2:
            break
    return buckets


def parse_bucket_probabilities(value: str) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for part in value.split(";"):
        pieces = part.strip().split()
        if len(pieces) < 2:
            continue
        probability_text = pieces[1].removesuffix("%")
        pairs.append((pieces[0], float(probability_text) / 100.0))
    return pairs


def reweighted_total_goal_buckets(value: str, factor_2_3: float) -> list[tuple[str, float]]:
    pairs = parse_bucket_probabilities(value)
    adjusted = [
        (bucket, probability * factor_2_3 if bucket == "2-3球" else probability)
        for bucket, probability in pairs
    ]
    total = sum(probability for _, probability in adjusted)
    if total <= 0:
        return pairs
    normalized = [(bucket, probability / total) for bucket, probability in adjusted]
    return sorted(normalized, key=lambda item: item[1], reverse=True)


def reweighted_bucket_hit(row: dict, factor_2_3: float) -> bool:
    buckets = reweighted_total_goal_buckets(row["raw_total_goal_buckets"], factor_2_3)
    return bool(buckets and buckets[0][0] == row["actual_total_bucket"])


def evaluated_row(result: dict, cache_run: dict, prediction: dict) -> dict:
    actual_score = result["actual_score"]
    actual = parse_score(actual_score)
    actual_outcome = outcome(actual)
    actual_bucket = total_bucket(sum(actual))
    final_score = result["final_score"]
    final_outcome = outcome(parse_score(final_score))
    selected_bucket = prediction["adjusted_total_goal_bucket"]
    selected_buckets = top2_buckets(prediction["adjusted_total_goals_top2"], selected_bucket)
    output = {
        "date_bjt": result["date_bjt"],
        "time_bjt": result["time_bjt"],
        "group": result["group"],
        "team_a": result["team_a"],
        "team_b": result["team_b"],
        "actual_score": actual_score,
        "regulation_score": result["regulation_score"],
        "final_score": final_score,
        "evaluation_phase": result["evaluation_phase"],
        "decision_method": result["decision_method"],
        "advancing_team": result["advancing_team"],
        "actual_outcome": actual_outcome,
        "final_outcome": final_outcome,
        "actual_total_bucket": actual_bucket,
        "cache_run_id": cache_run["run_id"],
        "cache_created_at_utc": cache_run["created_at_utc"].isoformat(),
        "predicted_outcome": prediction["predicted_outcome"],
        "outcome_hit": str(actual_outcome == prediction["predicted_outcome"]).upper(),
        "selected_total_bucket": selected_bucket,
        "selected_top2_buckets": ",".join(sorted(selected_buckets)),
        "raw_total_goal_buckets": prediction["adjusted_total_goals_top2"],
        "top1_bucket_hit": str(actual_bucket == selected_bucket).upper(),
        "top2_bucket_hit": str(actual_bucket in selected_buckets).upper(),
    }
    exact_hit = False
    score_outcome_hit = False
    bucket_hit = False
    final_score_outcome_hit = False
    deviations: list[float] = []
    missing_score_columns: list[str] = []
    for label, column in SCORE_COLUMNS:
        score = prediction_value(prediction, column)
        if score is None:
            missing_score_columns.append(label)
            output[f"{label}_score"] = ""
            output[f"{label}_exact_hit"] = ""
            output[f"{label}_outcome_hit"] = ""
            output[f"{label}_bucket_hit"] = ""
            output[f"{label}_deviation"] = ""
            continue
        score_tuple = parse_score(score)
        score_bucket = total_bucket(sum(score_tuple))
        deviation = score_deviation(actual_score, score)
        deviations.append(deviation)
        output[f"{label}_score"] = score
        output[f"{label}_exact_hit"] = str(score == actual_score).upper()
        output[f"{label}_outcome_hit"] = str(outcome(score_tuple) == actual_outcome).upper()
        output[f"{label}_final_outcome_hit"] = str(outcome(score_tuple) == final_outcome).upper()
        output[f"{label}_bucket_hit"] = str(score_bucket == actual_bucket).upper()
        output[f"{label}_deviation"] = f"{deviation:.6f}"
        exact_hit = exact_hit or score == actual_score
        score_outcome_hit = score_outcome_hit or outcome(score_tuple) == actual_outcome
        final_score_outcome_hit = final_score_outcome_hit or outcome(score_tuple) == final_outcome
        bucket_hit = bucket_hit or score_bucket == actual_bucket
    output["any_score_exact_hit"] = str(exact_hit).upper()
    output["any_score_outcome_hit"] = str(score_outcome_hit).upper()
    output["any_score_final_outcome_hit"] = str(final_score_outcome_hit).upper()
    output["any_score_bucket_hit"] = str(bucket_hit).upper()
    output["available_score_count"] = str(len(deviations))
    output["missing_score_columns"] = ",".join(missing_score_columns)
    output["four_score_mean_deviation"] = f"{sum(deviations) / len(deviations):.6f}" if deviations else ""
    return output


def bool_field(row: dict, key: str) -> bool:
    return row.get(key, "").upper() == "TRUE"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def summarize(rows: list[dict]) -> dict:
    deviations = [float(row["four_score_mean_deviation"]) for row in rows if row.get("four_score_mean_deviation")]
    return {
        "matches": len(rows),
        "outcome_hit": sum(bool_field(row, "outcome_hit") for row in rows),
        "top1_bucket_hit": sum(bool_field(row, "top1_bucket_hit") for row in rows),
        "top2_bucket_hit": sum(bool_field(row, "top2_bucket_hit") for row in rows),
        "any_exact_hit": sum(bool_field(row, "any_score_exact_hit") for row in rows),
        "any_score_bucket_hit": sum(bool_field(row, "any_score_bucket_hit") for row in rows),
        "mean_deviation": mean(deviations),
        "median_deviation": median(deviations),
    }


def score_column_summary(rows: list[dict], label: str) -> dict:
    available_rows = [row for row in rows if row.get(f"{label}_score")]
    deviations = [float(row[f"{label}_deviation"]) for row in available_rows]
    return {
        "available": len(available_rows),
        "exact": sum(bool_field(row, f"{label}_exact_hit") for row in available_rows),
        "outcome": sum(bool_field(row, f"{label}_outcome_hit") for row in available_rows),
        "bucket": sum(bool_field(row, f"{label}_bucket_hit") for row in available_rows),
        "mean_deviation": mean(deviations),
        "median_deviation": median(deviations),
    }


def write_detail(rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with DETAIL_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict], skipped: list[dict]) -> None:
    overall = summarize(rows)
    denominator = max(1, overall["matches"])
    lines = [
        "# Finished Match Evaluation From Realtime Cache",
        "",
        "只使用每场开赛前最后一次实时缓存。",
        "",
        "## Overall",
        "",
        "| Matches | Outcome hit | Top1 bucket hit | Top2 bucket hit | Any exact score | Any score bucket | Mean deviation | Median deviation |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {overall['matches']} | {pct(overall['outcome_hit'] / denominator)} | "
            f"{pct(overall['top1_bucket_hit'] / denominator)} | "
            f"{pct(overall['top2_bucket_hit'] / denominator)} | "
            f"{pct(overall['any_exact_hit'] / denominator)} | "
            f"{pct(overall['any_score_bucket_hit'] / denominator)} | "
            f"{overall['mean_deviation']:.3f} | {overall['median_deviation']:.3f} |"
        ),
        "",
        "## Score Columns",
        "",
        "| Score | Available | Exact | Outcome hit | Bucket hit | Mean deviation | Median deviation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _ in SCORE_COLUMNS:
        item = score_column_summary(rows, label)
        score_denominator = max(1, item["available"])
        lines.append(
            f"| {label} | {item['available']} | {pct(item['exact'] / score_denominator)} | "
            f"{pct(item['outcome'] / score_denominator)} | {pct(item['bucket'] / score_denominator)} | "
            f"{item['mean_deviation']:.3f} | {item['median_deviation']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Skipped",
            "",
            "| Date | Match | Reason |",
            "|---|---|---|",
        ]
    )
    for row in skipped:
        lines.append(f"| {row['date_bjt']} {row['time_bjt']} | {row['team_a']} vs {row['team_b']} | {row['reason']} |")
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_bucket_reweight_experiment(rows: list[dict]) -> None:
    denominator = max(1, len(rows))
    lines = [
        "# 2-3 Goal Bucket Reweight Experiment",
        "",
        "只削弱 `2-3球` 概率，再重新归一化并排序；不改其他桶。",
        "",
        "| 2-3 factor | Top1 hit | Hit rate |",
        "|---:|---:|---:|",
    ]
    for factor in TOTAL_GOAL_BUCKET_REWEIGHT_FACTORS:
        hits = sum(reweighted_bucket_hit(row, factor) for row in rows)
        lines.append(f"| {factor:.2f} | {hits}/{len(rows)} | {pct(hits / denominator)} |")
    BUCKET_REWEIGHT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    results = load_results()
    runs = load_cache_runs()
    rows: list[dict] = []
    skipped: list[dict] = []
    for result in results:
        found = cache_for_match(result, runs)
        if found is None:
            skipped.append({**result, "reason": "no pre-match realtime cache"})
            continue
        cache_run, prediction = found
        rows.append(evaluated_row(result, cache_run, prediction))
    write_detail(rows)
    write_summary(rows, skipped)
    write_bucket_reweight_experiment(rows)
    print(f"Detail CSV: {DETAIL_CSV}")
    print(f"Summary: {SUMMARY_MD}")
    print(f"Bucket reweight experiment: {BUCKET_REWEIGHT_MD}")
    print(f"Evaluated: {len(rows)}")
    print(f"Skipped: {len(skipped)}")


if __name__ == "__main__":
    main()
