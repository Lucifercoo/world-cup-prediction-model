from __future__ import annotations

import csv
import sys
from collections import Counter

from backtests.backtest_world_cup_fifa_profile_scores import (
    WORLD_CUPS,
    build_year_models,
    expected_goals,
    goals_per_match_by_stage,
    load_world_cup_matches,
    outcome_probabilities,
)
from backtests.backtest_world_cup_fifa_ranking import WorldCupMatch, stage_bucket
from predict_fifa_profile import (
    base_total_goal_bucket_from_expected,
    base_total_goal_bucket_probabilities_from_expected,
    expected_total_goals_value,
    second_bucket_from_expected_total_goals,
    total_goal_bucket,
)
from profiles import OUTPUT_DIR


MATCH_CSV = OUTPUT_DIR / "world_cup_continuous_total_goal_backtest_matches.csv"
SUMMARY_MD = OUTPUT_DIR / "world_cup_continuous_total_goal_backtest_summary.md"


def pct(value: float) -> str:
    return f"{value:.1%}"


def accuracy(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[key]) / len(rows)


def bucket_counts(rows: list[dict], field: str) -> Counter[str]:
    return Counter(row[field] for row in rows)


def predict_match(
    match: WorldCupMatch,
    model,
    base_goals_per_match_by_stage: dict[str, float],
) -> dict:
    p_home, p_draw, p_away = outcome_probabilities(match, model)
    lambda_home, lambda_away = expected_goals(match, model, base_goals_per_match_by_stage)
    expected_total = expected_total_goals_value(
        lambda_home,
        lambda_away,
        p_home,
        p_draw,
        p_away,
        model.rankings[match.home_team],
        model.rankings[match.away_team],
        model.profiles[match.home_team],
        model.profiles[match.away_team],
        model.baselines,
    )
    selected_bucket = base_total_goal_bucket_from_expected(expected_total)
    second_bucket = second_bucket_from_expected_total_goals(expected_total, selected_bucket)
    bucket_probabilities = base_total_goal_bucket_probabilities_from_expected(expected_total)
    actual_total = match.home_score + match.away_score
    actual_bucket = total_goal_bucket(actual_total)
    return {
        "year": match.year,
        "date": match.date.isoformat(),
        "stage": stage_bucket(match),
        "home_team": match.home_team,
        "away_team": match.away_team,
        "actual_score": f"{match.home_score}-{match.away_score}",
        "actual_total_goals": actual_total,
        "actual_total_goal_bucket": actual_bucket,
        "expected_total_goals": expected_total,
        "selected_total_goal_bucket": selected_bucket,
        "top2_total_goal_bucket": second_bucket,
        "top1_total_goal_bucket_hit": selected_bucket == actual_bucket,
        "top2_total_goal_bucket_hit": actual_bucket in {selected_bucket, second_bucket},
        "top_total_goal_buckets": "; ".join(
            f"{bucket} {probability:.1%}" for bucket, probability in bucket_probabilities
        ),
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "xg_home": lambda_home,
        "xg_away": lambda_away,
        "home_fifa_rank": model.rankings[match.home_team].rank,
        "away_fifa_rank": model.rankings[match.away_team].rank,
        "home_style": model.profiles[match.home_team].style,
        "away_style": model.profiles[match.away_team].style,
    }


def build_rows() -> tuple[list[dict], dict[int, list[dict]]]:
    matches = load_world_cup_matches()
    models = build_year_models(matches)
    all_years = set(WORLD_CUPS)
    rows: list[dict] = []
    rows_by_year: dict[int, list[dict]] = {}
    for target_year in sorted(WORLD_CUPS):
        training_years = all_years - {target_year}
        base_goals = goals_per_match_by_stage(matches, training_years)
        year_rows = [
            predict_match(match, models[target_year], base_goals)
            for match in matches
            if match.year == target_year
        ]
        rows.extend(year_rows)
        rows_by_year[target_year] = year_rows
    return rows, rows_by_year


def summary_row(label: str, rows: list[dict]) -> str:
    selected_2_3 = sum(1 for row in rows if row["selected_total_goal_bucket"] == "2-3球")
    actual_2_3 = sum(1 for row in rows if row["actual_total_goal_bucket"] == "2-3球")
    return (
        f"| {label} | {len(rows)} | "
        f"{sum(1 for row in rows if row['top1_total_goal_bucket_hit'])} | {pct(accuracy(rows, 'top1_total_goal_bucket_hit'))} | "
        f"{sum(1 for row in rows if row['top2_total_goal_bucket_hit'])} | {pct(accuracy(rows, 'top2_total_goal_bucket_hit'))} | "
        f"{selected_2_3} | {pct(selected_2_3 / len(rows)) if rows else '0.0%'} | "
        f"{actual_2_3} | {pct(actual_2_3 / len(rows)) if rows else '0.0%'} |"
    )


def write_match_csv(rows: list[dict]) -> None:
    fields = [
        "year",
        "date",
        "stage",
        "home_team",
        "away_team",
        "actual_score",
        "actual_total_goals",
        "actual_total_goal_bucket",
        "expected_total_goals",
        "selected_total_goal_bucket",
        "top2_total_goal_bucket",
        "top1_total_goal_bucket_hit",
        "top2_total_goal_bucket_hit",
        "top_total_goal_buckets",
        "p_home",
        "p_draw",
        "p_away",
        "xg_home",
        "xg_away",
        "home_fifa_rank",
        "away_fifa_rank",
        "home_style",
        "away_style",
    ]
    with MATCH_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row[field] for field in fields if field in row},
                    "expected_total_goals": f"{row['expected_total_goals']:.6f}",
                    "p_home": f"{row['p_home']:.6f}",
                    "p_draw": f"{row['p_draw']:.6f}",
                    "p_away": f"{row['p_away']:.6f}",
                    "xg_home": f"{row['xg_home']:.6f}",
                    "xg_away": f"{row['xg_away']:.6f}",
                }
            )


def write_summary(rows: list[dict], rows_by_year: dict[int, list[dict]]) -> None:
    selected_counts = bucket_counts(rows, "selected_total_goal_bucket")
    actual_counts = bucket_counts(rows, "actual_total_goal_bucket")
    lines = [
        "# 四届世界杯连续总进球桶回测",
        "",
        "- 覆盖年份：2010、2014、2018、2022。",
        "- 数据口径：世界杯开赛前 FIFA 排名、开赛前 10 年国家队画像、留一届验证的阶段总进球基准。",
        "- 方法：先算连续 `expected_total_goals`，再按阈值落到 `0-1球 / 2-3球 / 4-5球 / 6-8球`。",
        "- 不使用 2026 赛果、实时上下文、赛后技术统计或媒体评论。",
        "",
        "## 命中率",
        "",
        "| 范围 | 场次 | Top1命中 | Top1命中率 | Top2命中 | Top2命中率 | 选2-3 | 选2-3占比 | 实际2-3 | 实际2-3占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        summary_row("全部", rows),
    ]
    for year in sorted(rows_by_year):
        lines.append(summary_row(str(year), rows_by_year[year]))
    for stage in ["group", "knockout"]:
        label = "小组赛" if stage == "group" else "淘汰赛"
        lines.append(summary_row(label, [row for row in rows if row["stage"] == stage]))

    lines.extend(
        [
            "",
            "## 桶分布",
            "",
            "| 桶 | 预测Top1场次 | 预测Top1占比 | 实际场次 | 实际占比 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for bucket in ["0-1球", "2-3球", "4-5球", "6-8球"]:
        predicted_count = selected_counts[bucket]
        actual_count = actual_counts[bucket]
        lines.append(
            f"| {bucket} | {predicted_count} | {pct(predicted_count / len(rows))} | "
            f"{actual_count} | {pct(actual_count / len(rows))} |"
        )

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    rows, rows_by_year = build_rows()
    write_match_csv(rows)
    write_summary(rows, rows_by_year)
    print(f"Matches: {MATCH_CSV}")
    print(f"Summary: {SUMMARY_MD}")
    print(
        "Top1 total-goal bucket accuracy: "
        f"{sum(1 for row in rows if row['top1_total_goal_bucket_hit'])}/{len(rows)} = "
        f"{accuracy(rows, 'top1_total_goal_bucket_hit'):.1%}"
    )
    print(
        "Top2 total-goal bucket accuracy: "
        f"{sum(1 for row in rows if row['top2_total_goal_bucket_hit'])}/{len(rows)} = "
        f"{accuracy(rows, 'top2_total_goal_bucket_hit'):.1%}"
    )
    selected_2_3 = sum(1 for row in rows if row["selected_total_goal_bucket"] == "2-3球")
    actual_2_3 = sum(1 for row in rows if row["actual_total_goal_bucket"] == "2-3球")
    print(f"Selected 2-3 bucket: {selected_2_3}/{len(rows)} = {selected_2_3 / len(rows):.1%}")
    print(f"Actual 2-3 bucket: {actual_2_3}/{len(rows)} = {actual_2_3 / len(rows):.1%}")


if __name__ == "__main__":
    main()
