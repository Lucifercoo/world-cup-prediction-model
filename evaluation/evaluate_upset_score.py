from __future__ import annotations

import csv
import statistics
from pathlib import Path

from backtests import backtest_world_cup_fifa_profile_scores as history
from backtests.backtest_world_cup_fifa_ranking import (
    WORLD_CUPS,
    load_world_cup_matches,
    outcome,
)
from predict_fifa_profile import (
    best_score_inside_total_goal_buckets,
    outcome_adjusted_scores,
    predicted_outcome_from_probabilities,
    select_upset_or_compression_score,
    top_total_goal_buckets,
)
from prediction_rules import parse_score, score_outcome

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
HISTORY_CSV = OUTPUT_DIR / "upset_score_history_backtest.csv"
CURRENT_CSV = OUTPUT_DIR / "upset_score_2026_finished.csv"
SUMMARY_MD = OUTPUT_DIR / "upset_score_evaluation_summary.md"
REALTIME_PLAN_CSV = OUTPUT_DIR / "realtime_context_adjusted_plan.csv"
RESULTS_2026_CSV = ROOT / "data" / "world_cup_2026_results.csv"


def score_distance(actual: str, predicted: str) -> int:
    actual_home, actual_away = parse_score(actual)
    predicted_home, predicted_away = parse_score(predicted)
    return abs(actual_home - predicted_home) + abs(actual_away - predicted_away)


def score_deviation(actual: str, predicted: str) -> float:
    actual_home, actual_away = parse_score(actual)
    return score_distance(actual, predicted) / max(1, actual_home + actual_away)


def mean_score_deviation(actual: str, predictions: list[str]) -> float:
    return sum(score_deviation(actual, prediction) for prediction in predictions) / len(predictions)


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def pct(value: float) -> str:
    return f"{value:.1%}"


def summarize(rows: list[dict], score_key: str) -> dict:
    exact = sum(1 for row in rows if row[score_key] == row["actual_score"])
    deviation_values = [float(row[f"{score_key}_deviation"]) for row in rows]
    return {
        "matches": len(rows),
        "exact": exact,
        "exact_rate": exact / len(rows) if rows else 0.0,
        "mean_deviation": sum(deviation_values) / len(deviation_values) if deviation_values else 0.0,
        "median_deviation": median(deviation_values),
    }


def add_deviation_fields(row: dict, score_keys: list[str]) -> None:
    for key in score_keys:
        row[f"{key}_deviation"] = score_deviation(row["actual_score"], row[key])


def build_history_rows() -> list[dict]:
    matches = load_world_cup_matches()
    models = history.build_year_models(matches)
    rows: list[dict] = []
    for target_year in sorted(WORLD_CUPS):
        base_goals = history.goals_per_match_by_stage(matches, set(WORLD_CUPS) - {target_year})
        model = models[target_year]
        for match in [item for item in matches if item.year == target_year]:
            p_home, p_draw, p_away = history.outcome_probabilities(match, model)
            lambda_home, lambda_away = history.expected_goals(match, model, base_goals)
            cells = outcome_adjusted_scores(lambda_home, lambda_away, p_home, p_draw, p_away)
            predicted = predicted_outcome_from_probabilities(
                p_home,
                p_draw,
                p_away,
                home_label="home",
                draw_label="draw",
                away_label="away",
            )
            _, _, total_goals = history.select_recommended_score(cells, predicted)
            buckets = top_total_goal_buckets(total_goals)
            selected_buckets = {buckets[0][0], buckets[1][0]}
            model_score_cell = best_score_inside_total_goal_buckets(cells, {buckets[0][0]})
            aggressive_score_cell = best_score_inside_total_goal_buckets(cells, {buckets[1][0]})
            model_score = f"{model_score_cell[0]}-{model_score_cell[1]}"
            aggressive_score = f"{aggressive_score_cell[0]}-{aggressive_score_cell[1]}"
            excluded = {parse_score(model_score), parse_score(aggressive_score)}
            upset_score_cell = select_upset_or_compression_score(
                cells,
                selected_buckets,
                p_home,
                p_draw,
                p_away,
                excluded,
            )
            actual_score = f"{match.home_score}-{match.away_score}"
            actual_outcome = outcome(match.home_score, match.away_score)
            row = {
                "scope": "history",
                "year": match.year,
                "date": match.date.isoformat(),
                "stage": history.stage_bucket(match),
                "home_team": match.home_team,
                "away_team": match.away_team,
                "actual_score": actual_score,
                "actual_outcome": actual_outcome,
                "predicted_outcome": predicted,
                "outcome_correct": actual_outcome == predicted,
                "selected_bucket": buckets[0][0],
                "second_bucket": buckets[1][0],
                "model_score": model_score,
                "aggressive_score": aggressive_score,
                "upset_score": f"{upset_score_cell[0]}-{upset_score_cell[1]}",
                "p_home": p_home,
                "p_draw": p_draw,
                "p_away": p_away,
                "xg_home": lambda_home,
                "xg_away": lambda_away,
            }
            add_deviation_fields(row, ["model_score", "aggressive_score", "upset_score"])
            row["mean_deviation_old3"] = mean_score_deviation(actual_score, [model_score, aggressive_score, model_score])
            row["mean_deviation_new3"] = mean_score_deviation(actual_score, [model_score, aggressive_score, row["upset_score"]])
            rows.append(row)
    return rows


def load_2026_results() -> dict[tuple[str, str, str], str]:
    results: dict[tuple[str, str, str], str] = {}
    with RESULTS_2026_CSV.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["date_bjt"], row["team_a"], row["team_b"])
            results[key] = f"{int(row['goals_a'])}-{int(row['goals_b'])}"
    return results


def build_current_rows() -> list[dict]:
    results = load_2026_results()
    rows: list[dict] = []
    with REALTIME_PLAN_CSV.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["date_bjt"], row["team_a"], row["team_b"])
            if key not in results:
                continue
            p_a = float(row["adjusted_p_a"])
            p_draw = float(row["adjusted_p_draw"])
            p_b = float(row["adjusted_p_b"])
            lambda_a = float(row["adjusted_xg_a"])
            lambda_b = float(row["adjusted_xg_b"])
            cells = outcome_adjusted_scores(lambda_a, lambda_b, p_a, p_draw, p_b)
            selected_buckets = {row["adjusted_total_goal_bucket"]}
            for part in row["adjusted_total_goals_top2"].split(";"):
                bucket = part.strip().split(" ", maxsplit=1)[0]
                if bucket:
                    selected_buckets.add(bucket)
                if len(selected_buckets) >= 2:
                    break
            model_score = row["adjusted_score_1_model"]
            aggressive_score = row["adjusted_score_2_aggressive_prediction"]
            market_score = row["adjusted_score_3_market_value"]
            excluded = {parse_score(model_score), parse_score(aggressive_score)}
            upset_score_cell = select_upset_or_compression_score(
                cells,
                selected_buckets,
                p_a,
                p_draw,
                p_b,
                excluded,
            )
            actual_score = results[key]
            actual_home, actual_away = parse_score(actual_score)
            actual = score_outcome(actual_home, actual_away)
            current = {
                "scope": "2026_finished",
                "date": row["date_bjt"],
                "time": row["time_bjt"],
                "group": row["group"],
                "home_team": row["team_a"],
                "away_team": row["team_b"],
                "actual_score": actual_score,
                "actual_outcome": actual,
                "predicted_outcome": row["predicted_outcome"],
                "outcome_correct": actual == row["predicted_outcome"],
                "selected_bucket": row["adjusted_total_goal_bucket"],
                "second_bucket": next(bucket for bucket in selected_buckets if bucket != row["adjusted_total_goal_bucket"]),
                "model_score": model_score,
                "aggressive_score": aggressive_score,
                "market_score": market_score,
                "upset_score": f"{upset_score_cell[0]}-{upset_score_cell[1]}",
                "p_home": p_a,
                "p_draw": p_draw,
                "p_away": p_b,
                "xg_home": lambda_a,
                "xg_away": lambda_b,
            }
            add_deviation_fields(current, ["model_score", "aggressive_score", "market_score", "upset_score"])
            current["mean_deviation_old3"] = mean_score_deviation(actual_score, [model_score, aggressive_score, market_score])
            current["mean_deviation_new3"] = mean_score_deviation(actual_score, [model_score, aggressive_score, current["upset_score"]])
            rows.append(current)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exact_coverage(rows: list[dict], score_keys: list[str]) -> float:
    return sum(1 for row in rows if row["actual_score"] in {row[key] for key in score_keys}) / len(rows)


def mean_new_minus_old(rows: list[dict]) -> float:
    return sum(float(row["mean_deviation_new3"]) - float(row["mean_deviation_old3"]) for row in rows) / len(rows)


def favorite_probability(row: dict) -> float:
    return max(float(row["p_home"]), float(row["p_draw"]), float(row["p_away"]))


def probability_band_rows(rows: list[dict]) -> list[dict]:
    bands = [
        ("<=45%", 0.00, 0.45),
        ("45-55%", 0.45, 0.55),
        ("55-70%", 0.55, 0.70),
        (">70%", 0.70, 1.01),
    ]
    output = []
    for label, low, high in bands:
        selected = [row for row in rows if low <= favorite_probability(row) < high]
        if not selected:
            continue
        output.append(
            {
                "band": label,
                "matches": len(selected),
                "old_coverage": exact_coverage(
                    selected,
                    ["model_score", "aggressive_score", "market_score"]
                    if "market_score" in selected[0]
                    else ["model_score", "aggressive_score"],
                ),
                "new_coverage": exact_coverage(selected, ["model_score", "aggressive_score", "upset_score"]),
                "old_mean_deviation": sum(float(row["mean_deviation_old3"]) for row in selected) / len(selected),
                "new_mean_deviation": sum(float(row["mean_deviation_new3"]) for row in selected) / len(selected),
                "upset_exact": sum(1 for row in selected if row["actual_score"] == row["upset_score"]) / len(selected),
            }
        )
    return output


def write_summary(history_rows: list[dict], current_rows: list[dict]) -> None:
    history_model = summarize(history_rows, "model_score")
    history_aggressive = summarize(history_rows, "aggressive_score")
    history_upset = summarize(history_rows, "upset_score")
    current_model = summarize(current_rows, "model_score")
    current_aggressive = summarize(current_rows, "aggressive_score")
    current_market = summarize(current_rows, "market_score")
    current_upset = summarize(current_rows, "upset_score")

    lines = [
        "# Upset Score Evaluation",
        "",
        "## Summary",
        "",
        "| Scope | Matches | Model exact | Aggressive exact | Market exact | Upset exact | Old 3 exact coverage | New 3 exact coverage | New-old mean deviation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| 2010-2022 strict backtest | {len(history_rows)} | "
            f"{pct(history_model['exact_rate'])} | {pct(history_aggressive['exact_rate'])} | n/a | "
            f"{pct(history_upset['exact_rate'])} | "
            f"{pct(exact_coverage(history_rows, ['model_score', 'aggressive_score']))} | "
            f"{pct(exact_coverage(history_rows, ['model_score', 'aggressive_score', 'upset_score']))} | "
            f"{mean_new_minus_old(history_rows):.3f} |"
        ),
        (
            f"| 2026 finished diagnostic | {len(current_rows)} | "
            f"{pct(current_model['exact_rate'])} | {pct(current_aggressive['exact_rate'])} | "
            f"{pct(current_market['exact_rate'])} | {pct(current_upset['exact_rate'])} | "
            f"{pct(exact_coverage(current_rows, ['model_score', 'aggressive_score', 'market_score']))} | "
            f"{pct(exact_coverage(current_rows, ['model_score', 'aggressive_score', 'upset_score']))} | "
            f"{mean_new_minus_old(current_rows):.3f} |"
        ),
        "",
        "## Single Score Deviation",
        "",
        "| Scope | Score | Mean deviation | Median deviation |",
        "|---|---|---:|---:|",
        f"| 2010-2022 | model | {history_model['mean_deviation']:.3f} | {history_model['median_deviation']:.3f} |",
        f"| 2010-2022 | aggressive | {history_aggressive['mean_deviation']:.3f} | {history_aggressive['median_deviation']:.3f} |",
        f"| 2010-2022 | upset | {history_upset['mean_deviation']:.3f} | {history_upset['median_deviation']:.3f} |",
        f"| 2026 | model | {current_model['mean_deviation']:.3f} | {current_model['median_deviation']:.3f} |",
        f"| 2026 | aggressive | {current_aggressive['mean_deviation']:.3f} | {current_aggressive['median_deviation']:.3f} |",
        f"| 2026 | market | {current_market['mean_deviation']:.3f} | {current_market['median_deviation']:.3f} |",
        f"| 2026 | upset | {current_upset['mean_deviation']:.3f} | {current_upset['median_deviation']:.3f} |",
        "",
        "## Favorite Probability Bands",
        "",
        "| Scope | Favorite probability | Matches | Old exact coverage | New exact coverage | Old mean deviation | New mean deviation | Upset exact |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope, band_rows in [
        ("2010-2022", probability_band_rows(history_rows)),
        ("2026", probability_band_rows(current_rows)),
    ]:
        for row in band_rows:
            lines.append(
                f"| {scope} | {row['band']} | {row['matches']} | "
                f"{pct(row['old_coverage'])} | {pct(row['new_coverage'])} | "
                f"{row['old_mean_deviation']:.3f} | {row['new_mean_deviation']:.3f} | "
                f"{pct(row['upset_exact'])} |"
            )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    history_rows = build_history_rows()
    current_rows = build_current_rows()
    write_csv(HISTORY_CSV, history_rows)
    write_csv(CURRENT_CSV, current_rows)
    write_summary(history_rows, current_rows)
    print(f"History CSV: {HISTORY_CSV}")
    print(f"Current CSV: {CURRENT_CSV}")
    print(f"Summary: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
