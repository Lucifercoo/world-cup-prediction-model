from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from backtests.backtest_world_cup_fifa_ranking import (
    WORLD_CUPS,
    WorldCupMatch,
    load_ranking_snapshots,
    load_world_cup_matches,
    outcome,
    outcome_label,
    stage_bucket,
)
from predict_fifa_profile import (
    FifaRanking,
    ProfileBaselines,
    TeamProfile,
    best_score_inside_total_goal_buckets,
    clamp,
    draw_probability,
    outcome_adjusted_scores,
    outcome_uncertainty,
    best_score_for_total_goals,
    predicted_outcome_from_probabilities,
    profile_baselines,
    risk_label,
    risk_reasons,
    select_recommended_score,
    style_total_modifier,
    top_total_goal_buckets,
)
from profiles import OUTPUT_DIR, ProfileConfig, build_profiles, load_results_window


MATCH_CSV = OUTPUT_DIR / "world_cup_fifa_profile_score_backtest_matches.csv"
SUMMARY_MD = OUTPUT_DIR / "world_cup_fifa_profile_score_backtest_summary.md"
CALIBRATION_CSV = OUTPUT_DIR / "world_cup_fifa_profile_score_calibration.csv"
POINT_EDGE_SCALE = 180.0
HOST_EDGE_POINTS = 35.0
HOST_SPLIT_ADJUSTMENT = 0.03
CALIBRATION_BINS = [(0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.85)]


@dataclass(frozen=True)
class YearModel:
    year: int
    profile_start: date
    profile_cutoff: date
    fifa_snapshot_date: date
    rankings: dict[str, FifaRanking]
    profiles: dict[str, TeamProfile]
    baselines: ProfileBaselines


def pct(value: float) -> str:
    return f"{value:.1%}"


def year_teams(matches: list[WorldCupMatch], year: int) -> frozenset[str]:
    return frozenset(
        team
        for match in matches
        if match.year == year
        for team in (match.home_team, match.away_team)
    )


def build_rankings(year: int) -> tuple[dict[str, FifaRanking], date]:
    snapshot = load_ranking_snapshots()[year]
    rankings = {
        team: FifaRanking(
            rank=int(row["rank"]),
            points=float(row["total_points"]),
            snapshot_date=snapshot["snapshot_date"].isoformat(),
        )
        for team, row in snapshot["rankings"].items()
    }
    return rankings, snapshot["snapshot_date"]


def build_profiles_for_year(year: int, teams: frozenset[str], cutoff: date) -> dict[str, TeamProfile]:
    start = date(cutoff.year - 10, cutoff.month, cutoff.day)
    config = ProfileConfig(
        year=year,
        start_date=start,
        cutoff_date=cutoff,
        target_teams=teams,
        output_stem=f"scratch_team_profiles_{year}_score_backtest",
        target_label=f"{year} 世界杯 32 队",
    )
    rows = build_profiles(load_results_window(start, cutoff), config)
    return {
        row["team"]: TeamProfile(
            style=row["style"],
            goals_for=float(row["weighted_goals_for"]),
            goals_against=float(row["weighted_goals_against"]),
            clean_sheet_rate=float(row["clean_sheet_rate"]),
            multi_goal_rate=float(row["multi_goal_rate"]),
            conceded_multi_rate=float(row["conceded_multi_rate"]),
            high_total_goal_rate=float(row["high_total_goal_rate"]),
            both_score_rate=float(row["both_score_rate"]),
        )
        for row in rows
    }


def build_year_models(matches: list[WorldCupMatch]) -> dict[int, YearModel]:
    models: dict[int, YearModel] = {}
    for year in sorted(WORLD_CUPS):
        start_date, _ = WORLD_CUPS[year]
        cutoff = start_date - timedelta(days=1)
        teams = year_teams(matches, year)
        rankings, snapshot_date = build_rankings(year)
        profiles = build_profiles_for_year(year, teams, cutoff)
        missing_rankings = sorted(team for team in teams if team not in rankings)
        missing_profiles = sorted(team for team in teams if team not in profiles)
        if missing_rankings:
            raise RuntimeError(f"missing FIFA rankings for {year}: {', '.join(missing_rankings)}")
        if missing_profiles:
            raise RuntimeError(f"missing profiles for {year}: {', '.join(missing_profiles)}")
        models[year] = YearModel(
            year=year,
            profile_start=date(cutoff.year - 10, cutoff.month, cutoff.day),
            profile_cutoff=cutoff,
            fifa_snapshot_date=snapshot_date,
            rankings=rankings,
            profiles=profiles,
            baselines=profile_baselines(list(profiles.values())),
        )
    return models


def goals_per_match_by_stage(matches: list[WorldCupMatch], training_years: set[int]) -> dict[str, float]:
    totals: dict[str, list[int]] = defaultdict(list)
    for match in matches:
        if match.year in training_years:
            totals[stage_bucket(match)].append(match.home_score + match.away_score)
    missing = sorted({"group", "knockout"} - set(totals))
    if missing:
        raise RuntimeError(f"missing training stages for years {sorted(training_years)}: {', '.join(missing)}")
    return {stage: sum(values) / len(values) for stage, values in totals.items()}


def host_edge(team: str, match: WorldCupMatch) -> float:
    if match.year == 2010 and team == "South Africa":
        return HOST_EDGE_POINTS
    if match.year == 2014 and team == "Brazil":
        return HOST_EDGE_POINTS
    if match.year == 2018 and team == "Russia":
        return HOST_EDGE_POINTS
    if match.year == 2022 and team == "Qatar":
        return HOST_EDGE_POINTS
    return 0.0


def outcome_probabilities(match: WorldCupMatch, model: YearModel) -> tuple[float, float, float]:
    home_rank = model.rankings[match.home_team]
    away_rank = model.rankings[match.away_team]
    home_profile = model.profiles[match.home_team]
    away_profile = model.profiles[match.away_team]
    point_edge = home_rank.points - away_rank.points
    point_edge += host_edge(match.home_team, match)
    point_edge -= host_edge(match.away_team, match)
    non_draw_home = 1.0 / (1.0 + math.exp(-point_edge / POINT_EDGE_SCALE))
    p_draw = draw_probability(abs(home_rank.rank - away_rank.rank), home_profile, away_profile)
    non_draw_mass = 1.0 - p_draw
    return non_draw_mass * non_draw_home, p_draw, non_draw_mass * (1.0 - non_draw_home)


def expected_goals(
    match: WorldCupMatch,
    model: YearModel,
    base_goals_per_match_by_stage: dict[str, float],
) -> tuple[float, float]:
    home_rank = model.rankings[match.home_team]
    away_rank = model.rankings[match.away_team]
    home_profile = model.profiles[match.home_team]
    away_profile = model.profiles[match.away_team]
    p_home, _, p_away = outcome_probabilities(match, model)

    base_goals = base_goals_per_match_by_stage[stage_bucket(match)]
    total_goals = base_goals * style_total_modifier(home_profile, away_profile, model.baselines)
    attack_home = home_profile.goals_for / max(0.2, home_profile.goals_for + away_profile.goals_for)
    defense_away = away_profile.goals_against / max(0.2, home_profile.goals_against + away_profile.goals_against)
    split_home = 0.50 + (p_home - p_away) * 0.55
    split_home += (attack_home - 0.5) * 0.30
    split_home += (defense_away - 0.5) * 0.22
    split_home += (home_profile.multi_goal_rate - away_profile.conceded_multi_rate) * 0.08
    split_home -= (away_profile.clean_sheet_rate - home_profile.clean_sheet_rate) * 0.06
    split_home += (away_rank.rank - home_rank.rank) / 400.0
    split_home += HOST_SPLIT_ADJUSTMENT if host_edge(match.home_team, match) else 0.0
    split_home -= HOST_SPLIT_ADJUSTMENT if host_edge(match.away_team, match) else 0.0

    split_home = clamp(split_home, 0.18, 0.82)
    return (
        clamp(total_goals * split_home, 0.05, 4.2),
        clamp(total_goals * (1.0 - split_home), 0.05, 4.2),
    )


def top_outcome(predicted: str, p_home: float, p_draw: float, p_away: float) -> float:
    return {"home": p_home, "draw": p_draw, "away": p_away}[predicted]


def predict_match(
    match: WorldCupMatch,
    model: YearModel,
    base_goals_per_match_by_stage: dict[str, float],
) -> dict:
    p_home, p_draw, p_away = outcome_probabilities(match, model)
    lambda_home, lambda_away = expected_goals(match, model, base_goals_per_match_by_stage)
    cells = outcome_adjusted_scores(lambda_home, lambda_away, p_home, p_draw, p_away)
    predicted_outcome = predicted_outcome_from_probabilities(
        p_home,
        p_draw,
        p_away,
        home_label="home",
        draw_label="draw",
        away_label="away",
    )
    recommended, aligned_scores, total_goals = select_recommended_score(cells, predicted_outcome)
    total_goal_buckets = top_total_goal_buckets(total_goals)
    top1_total_goal_bucket = total_goal_buckets[0][0]
    top2_total_goal_bucket = total_goal_buckets[1][0]
    selected_total_goal_bucket_labels = {top1_total_goal_bucket}
    top2_total_goal_bucket_labels = {top1_total_goal_bucket, top2_total_goal_bucket}
    selected_total_goal_values = {total for total, _ in total_goals[:2]}
    total_constrained_score = best_score_for_total_goals(cells, selected_total_goal_values)
    bucket_primary_score = best_score_inside_total_goal_buckets(cells, selected_total_goal_bucket_labels)
    bucket_complement_score = best_score_inside_total_goal_buckets(cells, {top2_total_goal_bucket})
    top3_totals = {total for total, _ in total_goals[:3]}
    actual = outcome(match.home_score, match.away_score)
    actual_total = match.home_score + match.away_score
    actual_total_bucket = top_total_goal_buckets([(actual_total, 1.0)])[0][0]
    rank_gap = abs(model.rankings[match.home_team].rank - model.rankings[match.away_team].rank)
    uncertainty, _, margin = outcome_uncertainty(p_home, p_draw, p_away)
    return {
        "year": match.year,
        "date": match.date.isoformat(),
        "stage": stage_bucket(match),
        "home_team": match.home_team,
        "away_team": match.away_team,
        "actual_score": f"{match.home_score}-{match.away_score}",
        "actual_outcome": actual,
        "actual_outcome_label": outcome_label(actual),
        "actual_total_goals": actual_total,
        "predicted_outcome": predicted_outcome,
        "predicted_outcome_label": outcome_label(predicted_outcome),
        "outcome_correct": actual == predicted_outcome,
        "predicted_outcome_probability": top_outcome(predicted_outcome, p_home, p_draw, p_away),
        "recommended_score": f"{recommended[0]}-{recommended[1]}",
        "recommended_score_probability": recommended[2],
        "score_correct": match.home_score == recommended[0] and match.away_score == recommended[1],
        "recommended_total_goals": recommended[0] + recommended[1],
        "recommended_total_correct": actual_total == recommended[0] + recommended[1],
        "selected_total_goals": "-".join(str(total) for total in sorted(selected_total_goal_values)),
        "total_constrained_score": f"{total_constrained_score[0]}-{total_constrained_score[1]}",
        "total_constrained_score_probability": total_constrained_score[2],
        "total_constrained_score_correct": (
            match.home_score == total_constrained_score[0]
            and match.away_score == total_constrained_score[1]
        ),
        "bucket_primary_score": f"{bucket_primary_score[0]}-{bucket_primary_score[1]}",
        "bucket_primary_score_probability": bucket_primary_score[2],
        "bucket_primary_score_correct": (
            match.home_score == bucket_primary_score[0]
            and match.away_score == bucket_primary_score[1]
        ),
        "bucket_complement_score": f"{bucket_complement_score[0]}-{bucket_complement_score[1]}",
        "bucket_complement_score_probability": bucket_complement_score[2],
        "bucket_complement_score_correct": (
            match.home_score == bucket_complement_score[0]
            and match.away_score == bucket_complement_score[1]
        ),
        "mode_total_goals": total_goals[0][0],
        "mode_total_correct": actual_total == total_goals[0][0],
        "top2_total_hit": actual_total in {total for total, _ in total_goals[:2]},
        "top3_total_hit": actual_total in top3_totals,
        "top_total_goals": "; ".join(f"{total}球 {prob:.1%}" for total, prob in total_goals[:3]),
        "actual_total_goal_bucket": actual_total_bucket,
        "selected_total_goal_bucket": top1_total_goal_bucket,
        "top2_total_goal_bucket": top2_total_goal_bucket,
        "top1_total_goal_bucket_hit": actual_total_bucket in selected_total_goal_bucket_labels,
        "top2_total_goal_bucket_hit": actual_total_bucket in top2_total_goal_bucket_labels,
        "top_total_goal_buckets": "; ".join(f"{bucket} {prob:.1%}" for bucket, prob in total_goal_buckets),
        "top_scores": "; ".join(f"{home}-{away} {prob:.1%}" for home, away, prob in aligned_scores[:3]),
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "xg_home": lambda_home,
        "xg_away": lambda_away,
        "home_fifa_rank": model.rankings[match.home_team].rank,
        "away_fifa_rank": model.rankings[match.away_team].rank,
        "home_style": model.profiles[match.home_team].style,
        "away_style": model.profiles[match.away_team].style,
        "uncertainty_score": uncertainty,
        "outcome_margin": margin,
        "risk_label": risk_label(uncertainty),
        "risk_reasons": risk_reasons(p_home, p_draw, p_away, rank_gap, total_goals[0][0]),
    }


def accuracy(rows: list[dict], key: str) -> float:
    if not rows:
        raise ValueError("cannot compute accuracy of empty rows")
    return sum(1 for row in rows if row[key]) / len(rows)


def summary_row(label: str, rows: list[dict]) -> str:
    return (
        f"| {label} | {len(rows)} | "
        f"{sum(1 for row in rows if row['outcome_correct'])} | {pct(accuracy(rows, 'outcome_correct'))} | "
        f"{sum(1 for row in rows if row['score_correct'])} | {pct(accuracy(rows, 'score_correct'))} | "
        f"{sum(1 for row in rows if row['recommended_total_correct'])} | {pct(accuracy(rows, 'recommended_total_correct'))} | "
        f"{sum(1 for row in rows if row['top1_total_goal_bucket_hit'])} | {pct(accuracy(rows, 'top1_total_goal_bucket_hit'))} | "
        f"{sum(1 for row in rows if row['top2_total_goal_bucket_hit'])} | {pct(accuracy(rows, 'top2_total_goal_bucket_hit'))} | "
        f"{sum(1 for row in rows if row['top3_total_hit'])} | {pct(accuracy(rows, 'top3_total_hit'))} |"
    )


def calibration_rows(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for low, high in CALIBRATION_BINS:
        bucket = [
            row
            for row in rows
            if low <= float(row["predicted_outcome_probability"]) < high
        ]
        if not bucket:
            continue
        output.append(
            {
                "bucket": f"{low:.0%}-{high:.0%}",
                "matches": len(bucket),
                "avg_predicted_probability": sum(float(row["predicted_outcome_probability"]) for row in bucket) / len(bucket),
                "actual_accuracy": accuracy(bucket, "outcome_correct"),
                "calibration_error": abs(
                    sum(float(row["predicted_outcome_probability"]) for row in bucket) / len(bucket)
                    - accuracy(bucket, "outcome_correct")
                ),
            }
        )
    return output


def leave_one_cup_summary(rows_by_policy: dict[int, list[dict]]) -> list[dict]:
    output = []
    all_rows = [row for rows in rows_by_policy.values() for row in rows]
    for target_year in sorted(rows_by_policy):
        rows = rows_by_policy[target_year]
        output.append(
            {
                "year": target_year,
                "rows": rows,
                "outcome_accuracy": accuracy(rows, "outcome_correct"),
                "score_accuracy": accuracy(rows, "score_correct"),
                "top1_total_goal_bucket_accuracy": accuracy(rows, "top1_total_goal_bucket_hit"),
                "top2_total_goal_bucket_accuracy": accuracy(rows, "top2_total_goal_bucket_hit"),
                "top3_total_accuracy": accuracy(rows, "top3_total_hit"),
            }
        )
    output.append(
        {
            "year": "all",
            "rows": all_rows,
            "outcome_accuracy": accuracy(all_rows, "outcome_correct"),
            "score_accuracy": accuracy(all_rows, "score_correct"),
            "top1_total_goal_bucket_accuracy": accuracy(all_rows, "top1_total_goal_bucket_hit"),
            "top2_total_goal_bucket_accuracy": accuracy(all_rows, "top2_total_goal_bucket_hit"),
            "top3_total_accuracy": accuracy(all_rows, "top3_total_hit"),
        }
    )
    return output


def build_rows(matches: list[WorldCupMatch], models: dict[int, YearModel]) -> tuple[list[dict], dict[int, list[dict]]]:
    all_years = set(WORLD_CUPS)
    rows: list[dict] = []
    rows_by_target_year: dict[int, list[dict]] = {}
    for target_year in sorted(WORLD_CUPS):
        training_years = all_years - {target_year}
        base_goals = goals_per_match_by_stage(matches, training_years)
        year_rows = [
            predict_match(match, models[target_year], base_goals)
            for match in matches
            if match.year == target_year
        ]
        rows.extend(year_rows)
        rows_by_target_year[target_year] = year_rows
    return rows, rows_by_target_year


def write_match_csv(rows: list[dict]) -> None:
    fields = [
        "year",
        "date",
        "stage",
        "home_team",
        "away_team",
        "actual_score",
        "actual_outcome",
        "actual_outcome_label",
        "actual_total_goals",
        "predicted_outcome",
        "predicted_outcome_label",
        "outcome_correct",
        "predicted_outcome_probability",
        "recommended_score",
        "recommended_score_probability",
        "score_correct",
        "recommended_total_goals",
        "recommended_total_correct",
        "selected_total_goals",
        "total_constrained_score",
        "total_constrained_score_probability",
        "total_constrained_score_correct",
        "bucket_primary_score",
        "bucket_primary_score_probability",
        "bucket_primary_score_correct",
        "bucket_complement_score",
        "bucket_complement_score_probability",
        "bucket_complement_score_correct",
        "mode_total_goals",
        "mode_total_correct",
        "top2_total_hit",
        "top3_total_hit",
        "top_total_goals",
        "actual_total_goal_bucket",
        "selected_total_goal_bucket",
        "top2_total_goal_bucket",
        "top1_total_goal_bucket_hit",
        "top2_total_goal_bucket_hit",
        "top_total_goal_buckets",
        "top_scores",
        "p_home",
        "p_draw",
        "p_away",
        "xg_home",
        "xg_away",
        "home_fifa_rank",
        "away_fifa_rank",
        "home_style",
        "away_style",
        "uncertainty_score",
        "outcome_margin",
        "risk_label",
        "risk_reasons",
    ]
    with MATCH_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row[field] for field in fields if field in row},
                    "predicted_outcome_probability": f"{row['predicted_outcome_probability']:.6f}",
                    "recommended_score_probability": f"{row['recommended_score_probability']:.6f}",
                    "p_home": f"{row['p_home']:.6f}",
                    "p_draw": f"{row['p_draw']:.6f}",
                    "p_away": f"{row['p_away']:.6f}",
                    "xg_home": f"{row['xg_home']:.6f}",
                    "xg_away": f"{row['xg_away']:.6f}",
                    "total_constrained_score_probability": f"{row['total_constrained_score_probability']:.6f}",
                    "bucket_primary_score_probability": f"{row['bucket_primary_score_probability']:.6f}",
                    "uncertainty_score": f"{row['uncertainty_score']:.6f}",
                    "outcome_margin": f"{row['outcome_margin']:.6f}",
                }
            )


def write_calibration_csv(calibration: list[dict]) -> None:
    fields = ["bucket", "matches", "avg_predicted_probability", "actual_accuracy", "calibration_error"]
    with CALIBRATION_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in calibration:
            writer.writerow(
                {
                    "bucket": row["bucket"],
                    "matches": row["matches"],
                    "avg_predicted_probability": f"{row['avg_predicted_probability']:.6f}",
                    "actual_accuracy": f"{row['actual_accuracy']:.6f}",
                    "calibration_error": f"{row['calibration_error']:.6f}",
                }
            )


def write_summary(rows: list[dict], models: dict[int, YearModel], loo_rows: list[dict], calibration: list[dict]) -> None:
    by_stage = {
        "小组赛": [row for row in rows if row["stage"] == "group"],
        "淘汰赛": [row for row in rows if row["stage"] == "knockout"],
    }
    risk_counts = Counter(row["risk_label"] for row in rows)
    lines = [
        "# 四届世界杯 FIFA + 画像比分回测",
        "",
        "- 覆盖年份：2010、2014、2018、2022。",
        "- 每届画像窗口：世界杯开赛前一天往前 10 年。",
        "- 每届总进球基准：留一届验证，只用其他三届的小组赛/淘汰赛均值。",
        "- 金标准：世界杯正赛实际赛果、实际比分、实际总进球。",
        "- 不读取实时上下文、比赛形态、赛后技术统计、赛后媒体评论。",
        "",
        "## 数据窗口",
        "",
        "| 年份 | FIFA排名日期 | 画像窗口 |",
        "|---:|---|---|",
    ]
    for year, model in sorted(models.items()):
        lines.append(
            f"| {year} | {model.fifa_snapshot_date.isoformat()} | "
            f"{model.profile_start.isoformat()} 到 {model.profile_cutoff.isoformat()} |"
        )

    lines.extend(
        [
            "",
            "## 命中率",
            "",
            "| 范围 | 场次 | 赛果命中 | 命中率 | 精确比分命中 | 命中率 | 推荐总进球命中 | 命中率 | Top1总进球桶命中 | 命中率 | Top2总进球桶命中 | 命中率 | Top3总进球命中 | 命中率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            summary_row("全部", rows),
        ]
    )
    for year in sorted(WORLD_CUPS):
        lines.append(summary_row(str(year), [row for row in rows if row["year"] == year]))
    for label, stage_rows in by_stage.items():
        lines.append(summary_row(label, stage_rows))

    lines.extend(
        [
            "",
            "## 留一届验证",
            "",
            "| 验证年份 | 场次 | 赛果命中率 | 精确比分命中率 | Top1总进球桶命中率 | Top2总进球桶命中率 | Top3总进球命中率 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in loo_rows:
        label = row["year"] if row["year"] != "all" else "合计"
        lines.append(
            f"| {label} | {len(row['rows'])} | {pct(row['outcome_accuracy'])} | "
            f"{pct(row['score_accuracy'])} | {pct(row['top1_total_goal_bucket_accuracy'])} | "
            f"{pct(row['top2_total_goal_bucket_accuracy'])} | "
            f"{pct(row['top3_total_accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## 概率校准",
            "",
            "| 预测最高赛果概率 | 场次 | 平均预测概率 | 实际命中率 | 误差 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in calibration:
        lines.append(
            f"| {row['bucket']} | {row['matches']} | {pct(row['avg_predicted_probability'])} | "
            f"{pct(row['actual_accuracy'])} | {pct(row['calibration_error'])} |"
        )

    lines.extend(
        [
            "",
            "## 不确定性分布",
            "",
            "| 风险 | 场次 |",
            "|---|---:|",
        ]
    )
    for label in ["低", "中", "中高", "高"]:
        lines.append(f"| {label} | {risk_counts[label]} |")

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    matches = load_world_cup_matches()
    models = build_year_models(matches)
    rows, rows_by_target_year = build_rows(matches, models)
    loo = leave_one_cup_summary(rows_by_target_year)
    calibration = calibration_rows(rows)
    write_match_csv(rows)
    write_calibration_csv(calibration)
    write_summary(rows, models, loo, calibration)

    print(f"Matches: {MATCH_CSV}")
    print(f"Calibration: {CALIBRATION_CSV}")
    print(f"Summary: {SUMMARY_MD}")
    print(f"Outcome accuracy: {sum(1 for row in rows if row['outcome_correct'])}/{len(rows)} = {accuracy(rows, 'outcome_correct'):.1%}")
    print(f"Exact score accuracy: {sum(1 for row in rows if row['score_correct'])}/{len(rows)} = {accuracy(rows, 'score_correct'):.1%}")
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
    print(f"Top3 total-goals accuracy: {sum(1 for row in rows if row['top3_total_hit'])}/{len(rows)} = {accuracy(rows, 'top3_total_hit'):.1%}")


if __name__ == "__main__":
    main()
