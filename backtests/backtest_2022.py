from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from predict import RESULTS_CSV, canonical_team, download_results, parse_result_date
from profiles import (
    OUTPUT_DIR,
    ProfileConfig,
    build_profiles,
    load_results_window,
    write_csv,
    write_markdown,
)


START_DATE = date(2012, 11, 19)
CUTOFF_DATE = date(2022, 11, 19)
WORLD_CUP_START = date(2022, 11, 20)
WORLD_CUP_END = date(2022, 12, 18)
PROFILE_STEM = "team_profiles_2022_pre_world_cup"
MATCH_CSV = OUTPUT_DIR / "backtest_2022_matches.csv"
SUMMARY_MD = OUTPUT_DIR / "backtest_2022_summary.md"
MIN_STRENGTH_EDGE = 0.30
TARGET_ACCURACY = 0.70


@dataclass(frozen=True)
class BacktestMatch:
    date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    stage: str


def load_world_cup_2022_matches() -> list[BacktestMatch]:
    download_results()
    matches: list[BacktestMatch] = []
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            match_date = parse_result_date(row["date"])
            if match_date < WORLD_CUP_START or match_date > WORLD_CUP_END:
                continue
            if row.get("tournament") != "FIFA World Cup":
                continue
            if not row.get("home_score", "").isdigit() or not row.get("away_score", "").isdigit():
                continue
            matches.append(
                BacktestMatch(
                    date=match_date,
                    home_team=canonical_team(row["home_team"]),
                    away_team=canonical_team(row["away_team"]),
                    home_score=int(row["home_score"]),
                    away_score=int(row["away_score"]),
                    stage=row.get("stage", ""),
                )
            )
    matches.sort(key=lambda m: m.date)
    if len(matches) != 64:
        raise RuntimeError(f"expected 64 World Cup 2022 matches, got {len(matches)}")
    return matches


def outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home"
    if home_score < away_score:
        return "away"
    return "draw"


def predicted_outcomes(home_strength: float, away_strength: float) -> list[str]:
    edge = home_strength - away_strength
    if edge >= 0:
        picks = ["home"]
    else:
        picks = ["away"]
    if abs(edge) < MIN_STRENGTH_EDGE:
        picks.append("draw")
    return picks


def format_outcomes(values: list[str]) -> str:
    if not values:
        raise ValueError("predicted outcomes cannot be empty")
    return "/".join(outcome_label(value) for value in values)


def pick_count(rows: list[dict], count: int) -> list[dict]:
    return [row for row in rows if row["prediction_count"] == count]


def accuracy(rows: list[dict]) -> float:
    if not rows:
        raise ValueError("cannot compute accuracy of empty rows")
    return sum(1 for row in rows if row["correct"]) / len(rows)


def coverage(rows: list[dict]) -> float:
    if not rows:
        raise ValueError("cannot compute coverage of empty rows")
    return sum(row["prediction_count"] for row in rows) / (len(rows) * 3)


def average_prediction_count(rows: list[dict]) -> float:
    if not rows:
        raise ValueError("cannot compute average prediction count of empty rows")
    return sum(row["prediction_count"] for row in rows) / len(rows)


def outcome_label(value: str) -> str:
    return {"home": "主胜", "draw": "平", "away": "客胜"}[value]


def stage_bucket(match: BacktestMatch) -> str:
    if match.date <= date(2022, 12, 2):
        return "group"
    return "knockout"


def write_match_csv(rows: list[dict]) -> None:
    fields = [
        "date",
        "stage",
        "home_team",
        "away_team",
        "score",
        "actual_outcome",
        "predicted_outcomes",
        "prediction_count",
        "correct",
        "home_tier",
        "away_tier",
        "home_style",
        "away_style",
        "home_strength",
        "away_strength",
        "strength_diff",
    ]
    with MATCH_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict], ranked_profiles: list[dict], profile_config: ProfileConfig) -> None:
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    group_rows = [row for row in rows if row["stage"] == "group"]
    knockout_rows = [row for row in rows if row["stage"] == "knockout"]
    group_correct = sum(1 for row in group_rows if row["correct"])
    knockout_correct = sum(1 for row in knockout_rows if row["correct"])
    single_rows = pick_count(rows, 1)
    double_rows = pick_count(rows, 2)
    single_correct = sum(1 for row in single_rows if row["correct"])
    double_correct = sum(1 for row in double_rows if row["correct"])
    total_accuracy = accuracy(rows)
    status = "通过" if total_accuracy >= TARGET_ACCURACY else "未通过"

    lines = [
        "# 2022 世界杯画像回测",
        "",
        f"- 画像窗口：{profile_config.start_date.isoformat()} 到 {profile_config.cutoff_date.isoformat()}",
        f"- 金标准：{WORLD_CUP_START.isoformat()} 到 {WORLD_CUP_END.isoformat()} 2022 世界杯 64 场",
        f"- 预测规则：先预测强度高者胜；两队强度分差小于 {MIN_STRENGTH_EDGE:.2f} 时，再加入平局候选。",
        f"- 合格线：{TARGET_ACCURACY:.0%}；当前：{total_accuracy:.1%}，{status}。",
        f"- 候选覆盖率：{coverage(rows):.1%}。单场最多 3 种赛果，本模型平均每场给 {average_prediction_count(rows):.2f} 个候选。",
        "",
        "## 命中率",
        "",
        "| 范围 | 场次 | 命中 | 命中率 | 平均候选数 |",
        "|---|---:|---:|---:|---:|",
        f"| 全部 | {total} | {correct} | {correct / total:.1%} | {average_prediction_count(rows):.2f} |",
        f"| 小组赛 | {len(group_rows)} | {group_correct} | {group_correct / len(group_rows):.1%} | {average_prediction_count(group_rows):.2f} |",
        f"| 淘汰赛 | {len(knockout_rows)} | {knockout_correct} | {knockout_correct / len(knockout_rows):.1%} | {average_prediction_count(knockout_rows):.2f} |",
        f"| 单候选场次 | {len(single_rows)} | {single_correct} | {single_correct / len(single_rows):.1%} | 1.00 |",
        f"| 双候选场次 | {len(double_rows)} | {double_correct} | {double_correct / len(double_rows):.1%} | 2.00 |",
        "",
        "## 画像强度前 16",
        "",
        "| 排名 | 球队 | 档次 | 风格 | 强度分 |",
        "|---:|---|---|---|---:|",
    ]
    for rank, profile in enumerate(ranked_profiles[:16], start=1):
        lines.append(
            f"| {rank} | {profile['team']} | {profile['tier']} | {profile['style']} | {profile['strength_score']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 错误样例",
            "",
            "| 日期 | 比赛 | 实际 | 预测 | 强度差 |",
            "|---|---|---|---|---:|",
        ]
    )
    for row in [row for row in rows if not row["correct"]][:16]:
        lines.append(
            "| {date} | {home} {score} {away} | {actual} | {predicted} | {diff:.3f} |".format(
                date=row["date"],
                home=row["home_team"],
                score=row["score"],
                away=row["away_team"],
                actual=row["actual_outcome"],
                predicted=row["predicted_outcomes"],
                diff=row["strength_diff"],
            )
        )

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    matches = load_world_cup_2022_matches()
    teams = frozenset({team for m in matches for team in (m.home_team, m.away_team)})
    profile_config = ProfileConfig(
        year=2022,
        start_date=START_DATE,
        cutoff_date=CUTOFF_DATE,
        target_teams=teams,
        output_stem=PROFILE_STEM,
        target_label="2022 世界杯 32 队",
    )
    training_results = load_results_window(START_DATE, CUTOFF_DATE)
    ranked_profiles = build_profiles(training_results, profile_config)
    write_csv(ranked_profiles, profile_config)
    write_markdown({profile["team"]: profile for profile in ranked_profiles}, ranked_profiles, profile_config)
    profiles = {profile["team"]: profile for profile in ranked_profiles}

    rows: list[dict] = []
    for match in matches:
        home = profiles[match.home_team]
        away = profiles[match.away_team]
        actual = outcome(match.home_score, match.away_score)
        predicted = predicted_outcomes(home["strength_score"], away["strength_score"])
        rows.append(
            {
                "date": match.date.isoformat(),
                "stage": stage_bucket(match),
                "home_team": match.home_team,
                "away_team": match.away_team,
                "score": f"{match.home_score}-{match.away_score}",
                "actual_outcome": outcome_label(actual),
                "predicted_outcomes": format_outcomes(predicted),
                "prediction_count": len(predicted),
                "correct": actual in predicted,
                "home_tier": home["tier"],
                "away_tier": away["tier"],
                "home_style": home["style"],
                "away_style": away["style"],
                "home_strength": round(home["strength_score"], 6),
                "away_strength": round(away["strength_score"], 6),
                "strength_diff": round(home["strength_score"] - away["strength_score"], 6),
            }
        )

    write_match_csv(rows)
    write_summary(rows, ranked_profiles, profile_config)

    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    print(f"Profile: {profile_config.csv_path}")
    print(f"Matches: {MATCH_CSV}")
    print(f"Summary: {SUMMARY_MD}")
    print(f"Accuracy: {correct}/{total} = {correct / total:.1%}")
    print(f"Target: {TARGET_ACCURACY:.0%}")


if __name__ == "__main__":
    main()
