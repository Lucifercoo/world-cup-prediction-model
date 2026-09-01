from __future__ import annotations

import csv
import math
import sys
import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from predict import RESULTS_CSV, canonical_team, download_results, parse_result_date, schedule


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
DEFAULT_PROFILE_CUTOFF = date(2026, 6, 12)
ROLLING_YEARS = 10
HALF_LIFE_DAYS = 365.25 * 3
MAX_ITERATIONS = 8
MIN_COMPARISON_SAMPLE_SIZE = 12
MIN_COMPARISON_WEIGHTED_SAMPLE_SIZE = 6.0
OPPONENT_TIER_WEIGHTS = {
    "顶级": 1.35,
    "强": 1.15,
    "中": 1.00,
    "弱": 0.75,
}
MANUAL_RESULTS = [
    {
        "date": date(2026, 6, 11),
        "home_team": "Mexico",
        "away_team": "South Africa",
        "home_score": 2,
        "away_score": 0,
        "neutral": False,
        "tournament": "FIFA World Cup",
    },
    {
        "date": date(2026, 6, 11),
        "home_team": "South Korea",
        "away_team": "Czechia",
        "home_score": 2,
        "away_score": 1,
        "neutral": True,
        "tournament": "FIFA World Cup",
    },
]


@dataclass(frozen=True)
class ProfileConfig:
    year: int
    start_date: date
    cutoff_date: date
    target_teams: frozenset[str]
    output_stem: str
    target_label: str

    @property
    def csv_path(self) -> Path:
        return OUTPUT_DIR / f"{self.output_stem}.csv"

    @property
    def markdown_path(self) -> Path:
        return OUTPUT_DIR / f"{self.output_stem}.md"


FIELDNAMES = [
    "year",
    "team",
    "tier",
    "strength_score",
    "style",
    "weighted_points_per_match",
    "weighted_win_rate",
    "weighted_draw_rate",
    "weighted_loss_rate",
    "weighted_goals_for",
    "weighted_goals_against",
    "weighted_goal_diff",
    "clean_sheet_rate",
    "multi_goal_rate",
    "conceded_multi_rate",
    "high_total_goal_rate",
    "both_score_rate",
    "avg_opponent_strength_score",
    "sample_size",
    "weighted_sample_size",
    "iterations",
]


@dataclass
class Accumulator:
    sample_size: int = 0
    weighted_sample_size: float = 0.0
    points: float = 0.0
    wins: float = 0.0
    draws: float = 0.0
    losses: float = 0.0
    goals_for: float = 0.0
    goals_against: float = 0.0
    clean_sheets: float = 0.0
    multi_goals: float = 0.0
    conceded_multi: float = 0.0
    high_total_goals: float = 0.0
    both_scored: float = 0.0
    opponent_strength_sum: float = 0.0
    opponent_strength_weight: float = 0.0


def default_config(year: int = DEFAULT_PROFILE_CUTOFF.year) -> ProfileConfig:
    cutoff_date = DEFAULT_PROFILE_CUTOFF if year == DEFAULT_PROFILE_CUTOFF.year else date(year, 12, 31)
    start_date = date(cutoff_date.year - ROLLING_YEARS, cutoff_date.month, cutoff_date.day)
    return ProfileConfig(
        year=year,
        start_date=start_date,
        cutoff_date=cutoff_date,
        target_teams=frozenset(world_cup_team_set()),
        output_stem=f"team_profiles_{year}",
        target_label="世界杯 48 队",
    )


def load_results_window(start_date: date, cutoff_date: date) -> list[dict]:
    if start_date >= cutoff_date:
        raise ValueError("start_date must be before cutoff_date")
    download_results()
    rows: list[dict] = []
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            match_date = parse_result_date(row["date"])
            if match_date < start_date or match_date > cutoff_date:
                continue
            if not row.get("home_score", "").isdigit() or not row.get("away_score", "").isdigit():
                continue
            rows.append(
                {
                    "date": match_date,
                    "home_team": canonical_team(row["home_team"]),
                    "away_team": canonical_team(row["away_team"]),
                    "home_score": int(row["home_score"]),
                    "away_score": int(row["away_score"]),
                    "neutral": row.get("neutral", "FALSE").upper() == "TRUE",
                    "tournament": row.get("tournament", ""),
                }
            )
    existing = {
        (r["date"], r["home_team"], r["away_team"], r["home_score"], r["away_score"])
        for r in rows
    }
    for row in MANUAL_RESULTS:
        key = (row["date"], row["home_team"], row["away_team"], row["home_score"], row["away_score"])
        if start_date <= row["date"] <= cutoff_date and key not in existing:
            rows.append(row)
    rows.sort(key=lambda r: r["date"])
    return rows


def time_weight(match_date: date, cutoff_date: date) -> float:
    age_days = max(0, (cutoff_date - match_date).days)
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def safe_rate(value: float, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError("weighted denominator must be positive")
    return value / denominator


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute mean of empty list")
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def zscores(values_by_team: dict[str, float]) -> dict[str, float]:
    values = list(values_by_team.values())
    m = mean(values)
    s = stdev(values)
    if s == 0:
        return {team: 0.0 for team in values_by_team}
    return {team: (value - m) / s for team, value in values_by_team.items()}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty list")
    if pct < 0 or pct > 100:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def classify_style(profile: dict, thresholds: dict[str, float]) -> str:
    gf = profile["weighted_goals_for"]
    ga = profile["weighted_goals_against"]
    high_total = profile["high_total_goal_rate"]

    gf_high = gf >= thresholds["gf_p70"]
    gf_low = gf <= thresholds["gf_p30"]
    ga_low = ga <= thresholds["ga_p30"]
    ga_high = ga >= thresholds["ga_p70"]
    total_high = high_total >= thresholds["total_p70"]
    total_low = high_total <= thresholds["total_p30"]

    if gf_high and ga_low:
        return "攻守兼备型"
    if gf_high and ga_high and total_high:
        return "开放型"
    if gf_high:
        return "进攻型"
    if ga_low and total_low:
        return "防守型"
    if gf_low:
        return "低效型"
    return "均衡型"


def base_tier(rank_index: int, total: int) -> str:
    rank_ratio = (rank_index + 1) / total
    if rank_ratio <= 0.08:
        return "顶级"
    if rank_ratio <= 0.25:
        return "强"
    if rank_ratio <= 0.70:
        return "中"
    return "弱"


def apply_sample_protection(tier: str, sample_size: int, weighted_sample_size: float) -> str:
    protected = tier
    if sample_size < 12 and protected == "顶级":
        protected = "强"
    if weighted_sample_size < 6 and protected in {"顶级", "强"}:
        protected = "中"
    return protected


def add_observation(
    acc: Accumulator,
    weight: float,
    goals_for: int,
    goals_against: int,
) -> None:
    pts = points_for(goals_for, goals_against)
    total_goals = goals_for + goals_against

    acc.sample_size += 1
    acc.weighted_sample_size += weight
    acc.points += pts * weight
    acc.wins += (1 if pts == 3 else 0) * weight
    acc.draws += (1 if pts == 1 else 0) * weight
    acc.losses += (1 if pts == 0 else 0) * weight
    acc.goals_for += goals_for * weight
    acc.goals_against += goals_against * weight
    acc.clean_sheets += (1 if goals_against == 0 else 0) * weight
    acc.multi_goals += (1 if goals_for >= 2 else 0) * weight
    acc.conceded_multi += (1 if goals_against >= 2 else 0) * weight
    acc.high_total_goals += (1 if total_goals >= 3 else 0) * weight
    acc.both_scored += (1 if goals_for > 0 and goals_against > 0 else 0) * weight


def compute_raw_profiles(
    results: list[dict],
    config: ProfileConfig,
    tier_by_team: dict[str, str] | None = None,
) -> dict[str, dict]:
    accumulators: dict[str, Accumulator] = defaultdict(Accumulator)

    for row in results:
        home = row["home_team"]
        away = row["away_team"]
        home_score = row["home_score"]
        away_score = row["away_score"]
        base_weight = time_weight(row["date"], config.cutoff_date)
        home_weight = base_weight
        away_weight = base_weight
        if tier_by_team is not None:
            home_weight *= OPPONENT_TIER_WEIGHTS[tier_by_team.get(away, "中")]
            away_weight *= OPPONENT_TIER_WEIGHTS[tier_by_team.get(home, "中")]
        add_observation(accumulators[home], home_weight, home_score, away_score)
        add_observation(accumulators[away], away_weight, away_score, home_score)

    profiles: dict[str, dict] = {}
    for team, acc in accumulators.items():
        w = acc.weighted_sample_size
        goals_for = safe_rate(acc.goals_for, w)
        goals_against = safe_rate(acc.goals_against, w)
        profiles[team] = {
            "year": config.year,
            "team": team,
            "weighted_points_per_match": safe_rate(acc.points, w),
            "weighted_win_rate": safe_rate(acc.wins, w),
            "weighted_draw_rate": safe_rate(acc.draws, w),
            "weighted_loss_rate": safe_rate(acc.losses, w),
            "weighted_goals_for": goals_for,
            "weighted_goals_against": goals_against,
            "weighted_goal_diff": goals_for - goals_against,
            "clean_sheet_rate": safe_rate(acc.clean_sheets, w),
            "multi_goal_rate": safe_rate(acc.multi_goals, w),
            "conceded_multi_rate": safe_rate(acc.conceded_multi, w),
            "high_total_goal_rate": safe_rate(acc.high_total_goals, w),
            "both_score_rate": safe_rate(acc.both_scored, w),
            "avg_opponent_strength_score": 0.0,
            "sample_size": acc.sample_size,
            "weighted_sample_size": w,
            "iterations": 0,
        }
    return profiles


def world_cup_team_set() -> set[str]:
    return {team for teams in world_cup_groups().values() for team in teams}


def eligible_comparison_profiles(profiles: dict[str, dict]) -> dict[str, dict]:
    eligible = {
        team: profile
        for team, profile in profiles.items()
        if profile["sample_size"] >= MIN_COMPARISON_SAMPLE_SIZE
        and profile["weighted_sample_size"] >= MIN_COMPARISON_WEIGHTED_SAMPLE_SIZE
    }
    if len(eligible) < 48:
        raise RuntimeError(f"too few eligible teams for comparison: {len(eligible)}")
    return eligible


def target_profiles(profiles: dict[str, dict], config: ProfileConfig) -> dict[str, dict]:
    missing = sorted(team for team in config.target_teams if team not in profiles)
    if missing:
        raise RuntimeError(f"missing target team profiles: {', '.join(missing)}")
    return {team: profiles[team] for team in config.target_teams}


def compute_strength_scores(profiles: dict[str, dict]) -> None:
    comparable = eligible_comparison_profiles(profiles)
    ppg_z = zscores({team: p["weighted_points_per_match"] for team, p in comparable.items()})
    gd_z = zscores({team: p["weighted_goal_diff"] for team, p in comparable.items()})
    gf_z = zscores({team: p["weighted_goals_for"] for team, p in comparable.items()})
    ga_z = zscores({team: p["weighted_goals_against"] for team, p in comparable.items()})
    cs_z = zscores({team: p["clean_sheet_rate"] for team, p in comparable.items()})

    for team, profile in profiles.items():
        if team not in comparable:
            profile["strength_score"] = float("nan")
            continue
        profile["strength_score"] = (
            ppg_z[team] * 0.35
            + gd_z[team] * 0.30
            + gf_z[team] * 0.15
            - ga_z[team] * 0.15
            + cs_z[team] * 0.05
        )


def assign_tiers(profiles: dict[str, dict]) -> None:
    comparable = eligible_comparison_profiles(profiles)
    ranked = sorted(comparable.values(), key=lambda p: p["strength_score"], reverse=True)
    total = len(ranked)
    for index, profile in enumerate(ranked):
        tier = base_tier(index, total)
        profile["tier"] = apply_sample_protection(
            tier=tier,
            sample_size=profile["sample_size"],
            weighted_sample_size=profile["weighted_sample_size"],
        )
    for team, profile in profiles.items():
        if team not in comparable:
            profile["tier"] = "中"


def assign_target_output_tiers(profiles: dict[str, dict], config: ProfileConfig) -> None:
    ranked = sorted(
        target_profiles(profiles, config).values(),
        key=lambda p: p["strength_score"],
        reverse=True,
    )
    total = len(ranked)
    for index, profile in enumerate(ranked):
        tier = base_tier(index, total)
        profile["tier"] = apply_sample_protection(
            tier=tier,
            sample_size=profile["sample_size"],
            weighted_sample_size=profile["weighted_sample_size"],
        )


def assign_styles(profiles: dict[str, dict], config: ProfileConfig) -> None:
    comparable = target_profiles(profiles, config)
    thresholds = {
        "gf_p30": percentile([p["weighted_goals_for"] for p in comparable.values()], 30),
        "gf_p70": percentile([p["weighted_goals_for"] for p in comparable.values()], 70),
        "ga_p30": percentile([p["weighted_goals_against"] for p in comparable.values()], 30),
        "ga_p70": percentile([p["weighted_goals_against"] for p in comparable.values()], 70),
        "total_p30": percentile([p["high_total_goal_rate"] for p in comparable.values()], 30),
        "total_p70": percentile([p["high_total_goal_rate"] for p in comparable.values()], 70),
    }
    for profile in profiles.values():
        profile["style"] = classify_style(profile, thresholds) if profile["team"] in comparable else "未输出"


def compute_average_opponent_strength(
    results: list[dict],
    profiles: dict[str, dict],
    config: ProfileConfig,
) -> None:
    accumulators: dict[str, Accumulator] = defaultdict(Accumulator)

    for row in results:
        weight = time_weight(row["date"], config.cutoff_date)
        home = row["home_team"]
        away = row["away_team"]
        if home not in profiles or away not in profiles:
            raise KeyError(f"missing profile for match: {home} vs {away}")
        home_strength = profiles[home]["strength_score"]
        away_strength = profiles[away]["strength_score"]

        accumulators[home].opponent_strength_sum += away_strength * weight
        accumulators[home].opponent_strength_weight += weight
        accumulators[away].opponent_strength_sum += home_strength * weight
        accumulators[away].opponent_strength_weight += weight

    for team, profile in profiles.items():
        acc = accumulators[team]
        profile["avg_opponent_strength_score"] = safe_rate(
            acc.opponent_strength_sum,
            acc.opponent_strength_weight,
        )


def build_profiles(results: list[dict], config: ProfileConfig) -> list[dict]:
    profiles = compute_raw_profiles(results, config)
    compute_strength_scores(profiles)
    assign_tiers(profiles)

    previous_tiers = {team: profile["tier"] for team, profile in profiles.items()}
    seen_states: dict[tuple[tuple[str, str], ...], int] = {
        tuple(sorted(previous_tiers.items())): 0
    }
    iterations = 0
    for iteration in range(1, MAX_ITERATIONS + 1):
        profiles = compute_raw_profiles(results, config, previous_tiers)
        compute_strength_scores(profiles)
        assign_tiers(profiles)
        current_tiers = {team: profile["tier"] for team, profile in profiles.items()}
        changed = sum(
            1
            for team, tier in current_tiers.items()
            if previous_tiers.get(team) != tier
        )
        iterations = iteration
        if changed == 0:
            break
        state = tuple(sorted(current_tiers.items()))
        if state in seen_states:
            break
        seen_states[state] = iteration
        previous_tiers = current_tiers

    for profile in profiles.values():
        profile["iterations"] = iterations

    assign_target_output_tiers(profiles, config)
    assign_styles(profiles, config)
    compute_average_opponent_strength(results, profiles, config)
    return sorted(
        target_profiles(profiles, config).values(),
        key=lambda p: p["strength_score"],
        reverse=True,
    )


def world_cup_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for match in schedule():
        for team in (match.team_a, match.team_b):
            if team not in groups[match.group]:
                groups[match.group].append(team)
    return dict(sorted(groups.items()))


def assert_target_teams_have_profiles(profiles: dict[str, dict], config: ProfileConfig) -> None:
    missing = sorted(team for team in config.target_teams if team not in profiles)
    if missing:
        raise RuntimeError(f"missing target team profiles: {', '.join(missing)}")


def format_number(value: float) -> str:
    return f"{value:.3f}"


def write_csv(profiles: list[dict], config: ProfileConfig) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with config.csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for profile in profiles:
            writer.writerow({name: profile[name] for name in FIELDNAMES})


def write_markdown(
    profiles_by_team: dict[str, dict],
    ranked_profiles: list[dict],
    config: ProfileConfig,
) -> None:
    target_heading = "组" if config.target_teams == world_cup_team_set() else "排名"
    lines = [
        f"# {config.year} 国家队画像",
        "",
        f"- 数据窗口：{config.start_date.isoformat()} 到 {config.cutoff_date.isoformat()}",
        "- 包含时间权重和对手强度迭代修正。",
        "",
        f"## {config.target_label}",
        "",
        f"| {target_heading} | 球队 | 档次 | 风格 | 强度分 | 场均积分 | 进球 | 失球 | 净胜球 | 样本 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if config.target_teams == world_cup_team_set():
        for group, teams in world_cup_groups().items():
            for team in teams:
                p = profiles_by_team[team]
                lines.append(
                    "| {group} | {team} | {tier} | {style} | {score} | {ppg} | {gf} | {ga} | {gd} | {sample} |".format(
                        group=group,
                        team=team,
                        tier=p["tier"],
                        style=p["style"],
                        score=format_number(p["strength_score"]),
                        ppg=format_number(p["weighted_points_per_match"]),
                        gf=format_number(p["weighted_goals_for"]),
                        ga=format_number(p["weighted_goals_against"]),
                        gd=format_number(p["weighted_goal_diff"]),
                        sample=p["sample_size"],
                    )
                )
    else:
        for rank, p in enumerate(ranked_profiles, start=1):
            lines.append(
                "| {group} | {team} | {tier} | {style} | {score} | {ppg} | {gf} | {ga} | {gd} | {sample} |".format(
                    group=rank,
                    team=p["team"],
                    tier=p["tier"],
                    style=p["style"],
                    score=format_number(p["strength_score"]),
                    ppg=format_number(p["weighted_points_per_match"]),
                    gf=format_number(p["weighted_goals_for"]),
                    ga=format_number(p["weighted_goals_against"]),
                    gd=format_number(p["weighted_goal_diff"]),
                    sample=p["sample_size"],
                )
            )

    lines.extend(
        [
            "",
            "## 强度前 30",
            "",
            "| 排名 | 球队 | 档次 | 风格 | 强度分 | 场均积分 | 进球 | 失球 | 净胜球 |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, p in enumerate(ranked_profiles[:30], start=1):
        lines.append(
            "| {rank} | {team} | {tier} | {style} | {score} | {ppg} | {gf} | {ga} | {gd} |".format(
                rank=rank,
                team=p["team"],
                tier=p["tier"],
                style=p["style"],
                score=format_number(p["strength_score"]),
                ppg=format_number(p["weighted_points_per_match"]),
                gf=format_number(p["weighted_goals_for"]),
                ga=format_number(p["weighted_goals_against"]),
                gd=format_number(p["weighted_goal_diff"]),
            )
        )

    config.markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_profile(config: ProfileConfig) -> list[dict]:
    results = load_results_window(config.start_date, config.cutoff_date)
    ranked_profiles = build_profiles(results, config)
    profiles_by_team = {profile["team"]: profile for profile in ranked_profiles}
    assert_target_teams_have_profiles(profiles_by_team, config)
    write_csv(ranked_profiles, config)
    write_markdown(profiles_by_team, ranked_profiles, config)
    return ranked_profiles


def parse_date_arg(value: str) -> date:
    try:
        year, month, day = (int(part) for part in value.split("-"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def config_from_args(args: argparse.Namespace) -> ProfileConfig:
    year = args.year
    cutoff_date = args.cutoff_date or (
        DEFAULT_PROFILE_CUTOFF if year == DEFAULT_PROFILE_CUTOFF.year else date(year, 12, 31)
    )
    start_date = args.start_date or date(cutoff_date.year - ROLLING_YEARS, cutoff_date.month, cutoff_date.day)
    return ProfileConfig(
        year=year,
        start_date=start_date,
        cutoff_date=cutoff_date,
        target_teams=frozenset(world_cup_team_set()),
        output_stem=args.output_stem or f"team_profiles_{year}",
        target_label=args.target_label,
    )


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=DEFAULT_PROFILE_CUTOFF.year)
    parser.add_argument("--start-date", type=parse_date_arg)
    parser.add_argument("--cutoff-date", type=parse_date_arg)
    parser.add_argument("--output-stem")
    parser.add_argument("--target-label", default="世界杯 48 队")
    args = parser.parse_args()

    config = config_from_args(args)
    ranked_profiles = run_profile(config)

    print(f"Wrote {config.csv_path}")
    print(f"Wrote {config.markdown_path}")
    print("Top 10 teams:")
    for rank, profile in enumerate(ranked_profiles[:10], start=1):
        print(
            f"{rank:>2}. {profile['team']}: {profile['tier']} / {profile['style']} / "
            f"{profile['strength_score']:.3f}"
        )


if __name__ == "__main__":
    main()
