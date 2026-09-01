from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date

from backtest_world_cup_fifa_ranking import (
    WORLD_CUPS,
    load_ranking_snapshots,
    load_world_cup_matches,
    stage_bucket,
)
from predict import RESULTS_CSV, canonical_team, download_results, parse_result_date
from predict_fifa_profile import (
    DRAW_RANK_GAP,
    POINT_EDGE_SCALE,
    FifaRanking,
    ProfileBaselines,
    TeamProfile,
    clamp,
    draw_probability,
    outcome_adjusted_scores,
    profile_baselines,
    select_recommended_score,
    style_total_modifier,
)
from profiles import OUTPUT_DIR, ProfileConfig, build_profiles, load_results_window


YEAR = 2022
PROFILE_START = date(2012, 11, 19)
PROFILE_CUTOFF = date(2022, 11, 19)
WORLD_CUP_START = date(2022, 11, 20)
WORLD_CUP_END = date(2022, 12, 18)
MATCH_CSV = OUTPUT_DIR / "backtest_2022_fifa_profile_total_goals_matches.csv"
SUMMARY_MD = OUTPUT_DIR / "backtest_2022_fifa_profile_total_goals_summary.md"


@dataclass(frozen=True)
class BacktestMatch:
    date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    country: str
    round_index: int


def load_2022_matches() -> list[BacktestMatch]:
    download_results()
    matches: list[BacktestMatch] = []
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("tournament") != "FIFA World Cup":
                continue
            match_date = parse_result_date(row["date"])
            if match_date < WORLD_CUP_START or match_date > WORLD_CUP_END:
                continue
            if not row.get("home_score", "").isdigit() or not row.get("away_score", "").isdigit():
                raise RuntimeError(f"missing score for 2022 World Cup match: {row}")
            matches.append(
                BacktestMatch(
                    date=match_date,
                    home_team=canonical_team(row["home_team"]),
                    away_team=canonical_team(row["away_team"]),
                    home_score=int(row["home_score"]),
                    away_score=int(row["away_score"]),
                    country=canonical_team(row.get("country", "")),
                    round_index=0,
                )
            )
    matches.sort(key=lambda match: match.date)
    if len(matches) != 64:
        raise RuntimeError(f"expected 64 World Cup 2022 matches, got {len(matches)}")
    return [
        BacktestMatch(
            date=match.date,
            home_team=match.home_team,
            away_team=match.away_team,
            home_score=match.home_score,
            away_score=match.away_score,
            country=match.country,
            round_index=index,
        )
        for index, match in enumerate(matches, start=1)
    ]


def pre_2022_goals_per_match_by_stage() -> dict[str, float]:
    matches = load_world_cup_matches()
    totals_by_stage: dict[str, list[int]] = defaultdict(list)
    for match in matches:
        if match.year >= YEAR:
            continue
        totals_by_stage[stage_bucket(match)].append(match.home_score + match.away_score)
    missing = sorted({"group", "knockout"} - set(totals_by_stage))
    if missing:
        raise RuntimeError(f"missing pre-2022 World Cup stages: {', '.join(missing)}")
    return {
        stage: sum(values) / len(values)
        for stage, values in sorted(totals_by_stage.items())
    }


def build_2022_rankings() -> tuple[dict[str, FifaRanking], str]:
    snapshot = load_ranking_snapshots()[YEAR]
    rankings = {
        team: FifaRanking(
            rank=int(row["rank"]),
            points=float(row["total_points"]),
            snapshot_date=snapshot["snapshot_date"].isoformat(),
        )
        for team, row in snapshot["rankings"].items()
    }
    return rankings, snapshot["snapshot_date"].isoformat()


def build_2022_profiles(teams: frozenset[str]) -> dict[str, TeamProfile]:
    config = ProfileConfig(
        year=YEAR,
        start_date=PROFILE_START,
        cutoff_date=PROFILE_CUTOFF,
        target_teams=teams,
        output_stem="scratch_team_profiles_2022_total_goals",
        target_label="2022 世界杯 32 队",
    )
    ranked_profiles = build_profiles(load_results_window(PROFILE_START, PROFILE_CUTOFF), config)
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
        for row in ranked_profiles
    }


def host_edge(team: str, match: BacktestMatch) -> float:
    return 35.0 if team == match.country else 0.0


def host_split_adjustment(team: str, match: BacktestMatch) -> float:
    return 0.03 if team == match.country else 0.0


def outcome_probabilities(
    match: BacktestMatch,
    rankings: dict[str, FifaRanking],
    profiles: dict[str, TeamProfile],
) -> tuple[float, float, float]:
    home_rank = rankings[match.home_team]
    away_rank = rankings[match.away_team]
    home_profile = profiles[match.home_team]
    away_profile = profiles[match.away_team]
    point_edge = home_rank.points - away_rank.points
    point_edge += host_edge(match.home_team, match)
    point_edge -= host_edge(match.away_team, match)
    non_draw_home = 1.0 / (1.0 + math.exp(-point_edge / POINT_EDGE_SCALE))
    p_draw = draw_probability(abs(home_rank.rank - away_rank.rank), home_profile, away_profile)
    non_draw_mass = 1.0 - p_draw
    return non_draw_mass * non_draw_home, p_draw, non_draw_mass * (1.0 - non_draw_home)


def expected_goals(
    match: BacktestMatch,
    rankings: dict[str, FifaRanking],
    profiles: dict[str, TeamProfile],
    base_goals_per_match: float,
    baselines: ProfileBaselines,
) -> tuple[float, float]:
    home_rank = rankings[match.home_team]
    away_rank = rankings[match.away_team]
    home_profile = profiles[match.home_team]
    away_profile = profiles[match.away_team]
    p_home, _, p_away = outcome_probabilities(match, rankings, profiles)

    total_goals = base_goals_per_match * style_total_modifier(home_profile, away_profile, baselines)
    attack_home = home_profile.goals_for / max(0.2, home_profile.goals_for + away_profile.goals_for)
    defense_away = away_profile.goals_against / max(0.2, home_profile.goals_against + away_profile.goals_against)
    split_home = 0.50 + (p_home - p_away) * 0.55
    split_home += (attack_home - 0.5) * 0.30
    split_home += (defense_away - 0.5) * 0.22
    split_home += (home_profile.multi_goal_rate - away_profile.conceded_multi_rate) * 0.08
    split_home -= (away_profile.clean_sheet_rate - home_profile.clean_sheet_rate) * 0.06
    split_home += (away_rank.rank - home_rank.rank) / 400.0
    split_home += host_split_adjustment(match.home_team, match)
    split_home -= host_split_adjustment(match.away_team, match)

    split_home = clamp(split_home, 0.18, 0.82)
    lambda_home = total_goals * split_home
    lambda_away = total_goals * (1.0 - split_home)
    return clamp(lambda_home, 0.05, 4.2), clamp(lambda_away, 0.05, 4.2)


def predict_match(
    match: BacktestMatch,
    rankings: dict[str, FifaRanking],
    profiles: dict[str, TeamProfile],
    base_goals_per_match: float,
    baselines: ProfileBaselines,
) -> dict:
    p_home, p_draw, p_away = outcome_probabilities(match, rankings, profiles)
    lambda_home, lambda_away = expected_goals(match, rankings, profiles, base_goals_per_match, baselines)
    cells = outcome_adjusted_scores(lambda_home, lambda_away, p_home, p_draw, p_away)
    predicted_outcome = max({"home": p_home, "draw": p_draw, "away": p_away}, key={"home": p_home, "draw": p_draw, "away": p_away}.get)
    recommended, _, total_goals = select_recommended_score(cells, predicted_outcome)
    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "xg_home": lambda_home,
        "xg_away": lambda_away,
        "stage_base_goals_per_match": base_goals_per_match,
        "recommended_score": f"{recommended[0]}-{recommended[1]}",
        "recommended_total_goals": recommended[0] + recommended[1],
        "mode_total_goals": total_goals[0][0],
        "top_total_goals": total_goals,
    }


def pct(value: float) -> str:
    return f"{value:.1%}"


def accuracy(rows: list[dict], key: str) -> float:
    if not rows:
        raise ValueError("cannot compute accuracy of empty rows")
    return sum(1 for row in rows if row[key]) / len(rows)


def summary_row(label: str, rows: list[dict]) -> str:
    return (
        f"| {label} | {len(rows)} | "
        f"{sum(1 for row in rows if row['recommended_total_correct'])} | {pct(accuracy(rows, 'recommended_total_correct'))} | "
        f"{sum(1 for row in rows if row['mode_total_correct'])} | {pct(accuracy(rows, 'mode_total_correct'))} | "
        f"{sum(1 for row in rows if row['top2_total_hit'])} | {pct(accuracy(rows, 'top2_total_hit'))} | "
        f"{sum(1 for row in rows if row['top3_total_hit'])} | {pct(accuracy(rows, 'top3_total_hit'))} |"
    )


def write_match_csv(rows: list[dict]) -> None:
    fields = [
        "date",
        "stage",
        "home_team",
        "away_team",
        "actual_score",
        "actual_total_goals",
        "recommended_score",
        "recommended_total_goals",
        "recommended_total_correct",
        "mode_total_goals",
        "mode_total_correct",
        "top_total_goals",
        "top2_total_hit",
        "top3_total_hit",
        "p_home",
        "p_draw",
        "p_away",
        "xg_home",
        "xg_away",
        "stage_base_goals_per_match",
        "home_fifa_rank",
        "away_fifa_rank",
        "home_style",
        "away_style",
    ]
    with MATCH_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict], ranking_date: str, base_goals_per_match_by_stage: dict[str, float]) -> None:
    group_rows = [row for row in rows if row["stage"] == "group"]
    knockout_rows = [row for row in rows if row["stage"] == "knockout"]
    misses = [row for row in rows if not row["recommended_total_correct"]]
    actual_distribution = Counter(row["actual_total_goals"] for row in rows)
    predicted_distribution = Counter(row["recommended_total_goals"] for row in rows)

    lines = [
        "# 2022 世界杯总进球回测：FIFA排名 + 球队风格",
        "",
        f"- FIFA 排名日期：{ranking_date}",
        f"- 画像窗口：{PROFILE_START.isoformat()} 到 {PROFILE_CUTOFF.isoformat()}",
        "- 总进球基准：2010/2014/2018 世界杯分阶段均值，"
        f"小组赛 {base_goals_per_match_by_stage['group']:.3f} 球/场，"
        f"淘汰赛 {base_goals_per_match_by_stage['knockout']:.3f} 球/场。",
        "- 金标准：2022 世界杯每场实际总进球数。",
        "- 推荐比分总进球：当前输出给用户看的比分总进球。",
        "- 总进球众数：模型把所有比分概率按总进球数合并后的最高概率总进球。",
        "",
        "## 命中率",
        "",
        "| 范围 | 场次 | 推荐比分总进球命中 | 命中率 | 总进球众数命中 | 命中率 | Top2总进球命中 | 命中率 | Top3总进球命中 | 命中率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        summary_row("全部", rows),
        summary_row("小组赛", group_rows),
        summary_row("淘汰赛", knockout_rows),
        "",
        "## 总进球分布",
        "",
        "| 总进球 | 实际场次 | 推荐预测场次 |",
        "|---:|---:|---:|",
    ]
    for total_goals in sorted(set(actual_distribution) | set(predicted_distribution)):
        lines.append(
            f"| {total_goals} | {actual_distribution[total_goals]} | {predicted_distribution[total_goals]} |"
        )

    lines.extend(
        [
            "",
            "## 推荐总进球错误样例",
            "",
            "| 日期 | 比赛 | 实际总进球 | 推荐比分 | 推荐总进球 | Top3总进球 |",
            "|---|---|---:|---|---:|---|",
        ]
    )
    for row in misses[:20]:
        lines.append(
            f"| {row['date']} | {row['home_team']} {row['actual_score']} {row['away_team']} | "
            f"{row['actual_total_goals']} | {row['recommended_score']} | "
            f"{row['recommended_total_goals']} | {row['top_total_goals']} |"
        )

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_rows() -> tuple[list[dict], str, dict[str, float]]:
    matches = load_2022_matches()
    teams = frozenset(team for match in matches for team in (match.home_team, match.away_team))
    rankings, ranking_date = build_2022_rankings()
    profiles = build_2022_profiles(teams)
    missing_rankings = sorted(team for team in teams if team not in rankings)
    missing_profiles = sorted(team for team in teams if team not in profiles)
    if missing_rankings:
        raise RuntimeError(f"missing FIFA rankings: {', '.join(missing_rankings)}")
    if missing_profiles:
        raise RuntimeError(f"missing profiles: {', '.join(missing_profiles)}")

    base_goals_per_match_by_stage = pre_2022_goals_per_match_by_stage()
    baselines = profile_baselines(list(profiles.values()))
    rows: list[dict] = []
    for match in matches:
        stage = stage_bucket(match)
        base_goals_per_match = base_goals_per_match_by_stage[stage]
        prediction = predict_match(match, rankings, profiles, base_goals_per_match, baselines)
        actual_total = match.home_score + match.away_score
        top_totals = prediction["top_total_goals"]
        top2_totals = {total for total, _ in top_totals[:2]}
        top3_totals = {total for total, _ in top_totals[:3]}
        rows.append(
            {
                "date": match.date.isoformat(),
                "stage": stage,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "actual_score": f"{match.home_score}-{match.away_score}",
                "actual_total_goals": actual_total,
                "recommended_score": prediction["recommended_score"],
                "recommended_total_goals": prediction["recommended_total_goals"],
                "recommended_total_correct": actual_total == prediction["recommended_total_goals"],
                "mode_total_goals": prediction["mode_total_goals"],
                "mode_total_correct": actual_total == prediction["mode_total_goals"],
                "top_total_goals": "; ".join(f"{total}球 {prob:.1%}" for total, prob in top_totals[:3]),
                "top2_total_hit": actual_total in top2_totals,
                "top3_total_hit": actual_total in top3_totals,
                "p_home": f"{prediction['p_home']:.6f}",
                "p_draw": f"{prediction['p_draw']:.6f}",
                "p_away": f"{prediction['p_away']:.6f}",
                "xg_home": f"{prediction['xg_home']:.6f}",
                "xg_away": f"{prediction['xg_away']:.6f}",
                "stage_base_goals_per_match": f"{prediction['stage_base_goals_per_match']:.6f}",
                "home_fifa_rank": rankings[match.home_team].rank,
                "away_fifa_rank": rankings[match.away_team].rank,
                "home_style": profiles[match.home_team].style,
                "away_style": profiles[match.away_team].style,
            }
        )
    return rows, ranking_date, base_goals_per_match_by_stage


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    rows, ranking_date, base_goals_per_match_by_stage = build_rows()
    write_match_csv(rows)
    write_summary(rows, ranking_date, base_goals_per_match_by_stage)
    print(f"Matches: {MATCH_CSV}")
    print(f"Summary: {SUMMARY_MD}")
    print(
        "Recommended total-goals accuracy: "
        f"{sum(1 for row in rows if row['recommended_total_correct'])}/{len(rows)} = "
        f"{accuracy(rows, 'recommended_total_correct'):.1%}"
    )
    print(
        "Top3 total-goals accuracy: "
        f"{sum(1 for row in rows if row['top3_total_hit'])}/{len(rows)} = "
        f"{accuracy(rows, 'top3_total_hit'):.1%}"
    )


if __name__ == "__main__":
    main()
