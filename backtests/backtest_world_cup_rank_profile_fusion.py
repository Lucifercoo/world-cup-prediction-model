from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from backtests.backtest_world_cup_fifa_ranking import (
    DRAW_RANK_GAP,
    WORLD_CUPS,
    WorldCupMatch,
    format_outcomes,
    load_ranking_snapshots,
    load_world_cup_matches,
    outcome,
    outcome_label,
    stage_bucket,
)
from profiles import OUTPUT_DIR, ProfileConfig, build_profiles, load_results_window


GRID_CSV = OUTPUT_DIR / "world_cup_rank_profile_fusion_grid.csv"
MATCH_CSV = OUTPUT_DIR / "world_cup_rank_profile_fusion_matches.csv"
SUMMARY_MD = OUTPUT_DIR / "world_cup_rank_profile_fusion_summary.md"
RAW_WEIGHTS = [round(value / 20, 2) for value in range(0, 21)]
RESIDUAL_ALPHAS = [round(value / 20, 2) for value in range(0, 21)]


@dataclass(frozen=True)
class YearInputs:
    year: int
    profile_start: date
    profile_cutoff: date
    fifa_snapshot_date: date
    fifa_rank: dict[str, int]
    fifa_points: dict[str, float]
    fifa_z: dict[str, float]
    profile_z: dict[str, float]
    profile_residual_z: dict[str, float]


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute mean of empty list")
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("cannot compute stdev of fewer than 2 values")
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def zscores(values_by_team: dict[str, float]) -> dict[str, float]:
    avg = mean(list(values_by_team.values()))
    sd = stdev(list(values_by_team.values()))
    if sd == 0:
        raise ValueError("cannot z-score constant values")
    return {team: (value - avg) / sd for team, value in values_by_team.items()}


def residualize_profile(fifa_z: dict[str, float], profile_z: dict[str, float]) -> dict[str, float]:
    teams = sorted(fifa_z)
    numerator = sum(fifa_z[team] * profile_z[team] for team in teams)
    denominator = sum(fifa_z[team] ** 2 for team in teams)
    if denominator == 0:
        raise ValueError("cannot residualize against zero FIFA variance")
    beta = numerator / denominator
    residuals = {team: profile_z[team] - beta * fifa_z[team] for team in teams}
    return zscores(residuals)


def year_target_teams(matches: list[WorldCupMatch], year: int) -> frozenset[str]:
    return frozenset(
        team
        for match in matches
        if match.year == year
        for team in (match.home_team, match.away_team)
    )


def build_year_inputs(matches: list[WorldCupMatch], snapshots: dict[int, dict]) -> dict[int, YearInputs]:
    inputs: dict[int, YearInputs] = {}
    for year in sorted(WORLD_CUPS):
        world_cup_start, _ = WORLD_CUPS[year]
        profile_cutoff = world_cup_start - timedelta(days=1)
        profile_start = date(profile_cutoff.year - 10, profile_cutoff.month, profile_cutoff.day)
        teams = year_target_teams(matches, year)
        snapshot = snapshots[year]
        rankings = snapshot["rankings"]
        missing_rankings = sorted(team for team in teams if team not in rankings)
        if missing_rankings:
            raise RuntimeError(f"missing FIFA rankings for {year}: {', '.join(missing_rankings)}")

        config = ProfileConfig(
            year=year,
            start_date=profile_start,
            cutoff_date=profile_cutoff,
            target_teams=teams,
            output_stem=f"scratch_team_profiles_{year}",
            target_label=f"{year} 世界杯 32 队",
        )
        profiles = {
            profile["team"]: profile
            for profile in build_profiles(load_results_window(profile_start, profile_cutoff), config)
        }
        missing_profiles = sorted(team for team in teams if team not in profiles)
        if missing_profiles:
            raise RuntimeError(f"missing profiles for {year}: {', '.join(missing_profiles)}")

        profile_strength = {team: float(profiles[team]["strength_score"]) for team in teams}
        nonfinite = sorted(team for team, value in profile_strength.items() if not math.isfinite(value))
        if nonfinite:
            raise RuntimeError(f"non-finite profile strength for {year}: {', '.join(nonfinite)}")

        fifa_points = {team: float(rankings[team]["total_points"]) for team in teams}
        fifa_rank = {team: int(rankings[team]["rank"]) for team in teams}
        fifa_z = zscores(fifa_points)
        profile_z = zscores(profile_strength)
        inputs[year] = YearInputs(
            year=year,
            profile_start=profile_start,
            profile_cutoff=profile_cutoff,
            fifa_snapshot_date=snapshot["snapshot_date"],
            fifa_rank=fifa_rank,
            fifa_points=fifa_points,
            fifa_z=fifa_z,
            profile_z=profile_z,
            profile_residual_z=residualize_profile(fifa_z, profile_z),
        )
    return inputs


def model_score(team: str, inputs: YearInputs, model: str, value: float) -> float:
    if model == "rank":
        return inputs.fifa_z[team]
    if model == "raw_blend":
        return inputs.fifa_z[team] * (1 - value) + inputs.profile_z[team] * value
    if model == "residual":
        return inputs.fifa_z[team] + inputs.profile_residual_z[team] * value
    raise ValueError(f"unknown model: {model}")


def predicted_outcomes(match: WorldCupMatch, inputs: YearInputs, model: str, value: float) -> list[str]:
    home_score = model_score(match.home_team, inputs, model, value)
    away_score = model_score(match.away_team, inputs, model, value)
    if home_score > away_score:
        picks = ["home"]
    elif home_score < away_score:
        picks = ["away"]
    else:
        picks = ["draw"]
    rank_gap = abs(inputs.fifa_rank[match.home_team] - inputs.fifa_rank[match.away_team])
    if rank_gap <= DRAW_RANK_GAP and "draw" not in picks:
        picks.append("draw")
    return picks


def evaluate(
    matches: list[WorldCupMatch],
    inputs_by_year: dict[int, YearInputs],
    model: str,
    value: float,
    years: set[int] | None = None,
) -> dict:
    rows = []
    for match in matches:
        if years is not None and match.year not in years:
            continue
        inputs = inputs_by_year[match.year]
        actual = outcome(match.home_score, match.away_score)
        predicted = predicted_outcomes(match, inputs, model, value)
        rows.append(
            {
                "match": match,
                "actual": actual,
                "predicted": predicted,
                "correct": actual in predicted,
            }
        )
    correct = sum(1 for row in rows if row["correct"])
    return {
        "model": model,
        "value": value,
        "rows": rows,
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "average_candidates": sum(len(row["predicted"]) for row in rows) / len(rows),
    }


def best_grid_result(
    matches: list[WorldCupMatch],
    inputs_by_year: dict[int, YearInputs],
    model: str,
    values: list[float],
    years: set[int],
) -> dict:
    results = [evaluate(matches, inputs_by_year, model, value, years) for value in values]
    return sorted(results, key=lambda row: (-row["accuracy"], row["value"]))[0]


def leave_one_cup_out(
    matches: list[WorldCupMatch],
    inputs_by_year: dict[int, YearInputs],
    model: str,
    values: list[float],
) -> dict:
    heldout_rows = []
    choices = []
    years = set(WORLD_CUPS)
    for target_year in sorted(WORLD_CUPS):
        train_years = years - {target_year}
        best = best_grid_result(matches, inputs_by_year, model, values, train_years)
        test = evaluate(matches, inputs_by_year, model, best["value"], {target_year})
        choices.append(
            {
                "target_year": target_year,
                "chosen_value": best["value"],
                "train_accuracy": best["accuracy"],
                "test_accuracy": test["accuracy"],
                "test_correct": test["correct"],
                "test_total": test["total"],
            }
        )
        heldout_rows.extend(test["rows"])
    correct = sum(1 for row in heldout_rows if row["correct"])
    return {
        "model": model,
        "choices": choices,
        "total": len(heldout_rows),
        "correct": correct,
        "accuracy": correct / len(heldout_rows),
        "average_candidates": sum(len(row["predicted"]) for row in heldout_rows) / len(heldout_rows),
    }


def write_grid_csv(rows: list[dict]) -> None:
    fields = ["model", "value", "total", "correct", "accuracy", "average_candidates"]
    with GRID_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "value": f"{row['value']:.2f}",
                    "total": row["total"],
                    "correct": row["correct"],
                    "accuracy": f"{row['accuracy']:.6f}",
                    "average_candidates": f"{row['average_candidates']:.6f}",
                }
            )


def write_match_csv(rows: list[dict], inputs_by_year: dict[int, YearInputs]) -> None:
    fields = [
        "year",
        "date",
        "stage",
        "home_team",
        "away_team",
        "score",
        "actual_outcome",
        "predicted_outcomes",
        "prediction_count",
        "correct",
        "home_fifa_rank",
        "away_fifa_rank",
        "home_fifa_z",
        "away_fifa_z",
        "home_profile_z",
        "away_profile_z",
        "home_profile_residual_z",
        "away_profile_residual_z",
    ]
    with MATCH_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            match = row["match"]
            inputs = inputs_by_year[match.year]
            writer.writerow(
                {
                    "year": match.year,
                    "date": match.date.isoformat(),
                    "stage": stage_bucket(match),
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "score": f"{match.home_score}-{match.away_score}",
                    "actual_outcome": outcome_label(row["actual"]),
                    "predicted_outcomes": format_outcomes(row["predicted"]),
                    "prediction_count": len(row["predicted"]),
                    "correct": row["correct"],
                    "home_fifa_rank": inputs.fifa_rank[match.home_team],
                    "away_fifa_rank": inputs.fifa_rank[match.away_team],
                    "home_fifa_z": f"{inputs.fifa_z[match.home_team]:.6f}",
                    "away_fifa_z": f"{inputs.fifa_z[match.away_team]:.6f}",
                    "home_profile_z": f"{inputs.profile_z[match.home_team]:.6f}",
                    "away_profile_z": f"{inputs.profile_z[match.away_team]:.6f}",
                    "home_profile_residual_z": f"{inputs.profile_residual_z[match.home_team]:.6f}",
                    "away_profile_residual_z": f"{inputs.profile_residual_z[match.away_team]:.6f}",
                }
            )


def pct(value: float) -> str:
    return f"{value:.1%}"


def result_row(label: str, row: dict) -> str:
    value = "-" if row["model"] == "rank" else f"{row['value']:.2f}"
    return (
        f"| {label} | {value} | {row['total']} | {row['correct']} | "
        f"{pct(row['accuracy'])} | {row['average_candidates']:.2f} |"
    )


def write_summary(
    inputs_by_year: dict[int, YearInputs],
    rank_result: dict,
    raw_best: dict,
    residual_best: dict,
    raw_loo: dict,
    residual_loo: dict,
) -> None:
    lines = [
        "# 世界杯排名 + 10年画像融合回测",
        "",
        "- 覆盖年份：2010、2014、2018、2022。",
        "- 画像窗口：每届世界杯开赛前一天往前 10 年。",
        "- 金标准：对应年份世界杯 64 场正赛赛果。",
        f"- 平局候选规则：FIFA 排名差 <= {DRAW_RANK_GAP} 时加入平局。",
        "",
        "## 数据窗口",
        "",
        "| 年份 | FIFA排名日期 | 画像窗口 |",
        "|---:|---|---|",
    ]
    for year in sorted(inputs_by_year):
        row = inputs_by_year[year]
        lines.append(
            f"| {year} | {row.fifa_snapshot_date.isoformat()} | "
            f"{row.profile_start.isoformat()} 到 {row.profile_cutoff.isoformat()} |"
        )

    lines.extend(
        [
            "",
            "## 合并结果",
            "",
            "| 模型 | 参数 | 场次 | 命中 | 命中率 | 平均候选数 |",
            "|---|---:|---:|---:|---:|---:|",
            result_row("纯 FIFA 排名", rank_result),
            result_row("FIFA + 原始画像，四届内最佳", raw_best),
            result_row("FIFA + 画像残差，四届内最佳", residual_best),
            f"| FIFA + 原始画像，留一届验证 | - | {raw_loo['total']} | {raw_loo['correct']} | {pct(raw_loo['accuracy'])} | {raw_loo['average_candidates']:.2f} |",
            f"| FIFA + 画像残差，留一届验证 | - | {residual_loo['total']} | {residual_loo['correct']} | {pct(residual_loo['accuracy'])} | {residual_loo['average_candidates']:.2f} |",
            "",
            "## 留一届验证参数",
            "",
            "| 模型 | 验证年份 | 训练选中参数 | 训练命中率 | 验证命中率 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, loo in [("原始画像", raw_loo), ("画像残差", residual_loo)]:
        for choice in loo["choices"]:
            lines.append(
                f"| {label} | {choice['target_year']} | {choice['chosen_value']:.2f} | "
                f"{pct(choice['train_accuracy'])} | {pct(choice['test_accuracy'])} |"
            )

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    matches = load_world_cup_matches()
    snapshots = load_ranking_snapshots()
    inputs_by_year = build_year_inputs(matches, snapshots)
    all_years = set(WORLD_CUPS)

    rank_result = evaluate(matches, inputs_by_year, "rank", 0.0, all_years)
    raw_grid = [evaluate(matches, inputs_by_year, "raw_blend", value, all_years) for value in RAW_WEIGHTS]
    residual_grid = [evaluate(matches, inputs_by_year, "residual", value, all_years) for value in RESIDUAL_ALPHAS]
    raw_best = sorted(raw_grid, key=lambda row: (-row["accuracy"], row["value"]))[0]
    residual_best = sorted(residual_grid, key=lambda row: (-row["accuracy"], row["value"]))[0]
    raw_loo = leave_one_cup_out(matches, inputs_by_year, "raw_blend", RAW_WEIGHTS)
    residual_loo = leave_one_cup_out(matches, inputs_by_year, "residual", RESIDUAL_ALPHAS)

    write_grid_csv([rank_result, *raw_grid, *residual_grid])
    write_match_csv(residual_best["rows"], inputs_by_year)
    write_summary(inputs_by_year, rank_result, raw_best, residual_best, raw_loo, residual_loo)

    print(f"Grid: {GRID_CSV}")
    print(f"Matches: {MATCH_CSV}")
    print(f"Summary: {SUMMARY_MD}")
    print(f"Rank: {rank_result['correct']}/{rank_result['total']} = {rank_result['accuracy']:.1%}")
    print(f"Raw blend best: weight={raw_best['value']:.2f}, accuracy={raw_best['accuracy']:.1%}")
    print(f"Residual best: alpha={residual_best['value']:.2f}, accuracy={residual_best['accuracy']:.1%}")
    print(f"Raw LOO: {raw_loo['accuracy']:.1%}")
    print(f"Residual LOO: {residual_loo['accuracy']:.1%}")


if __name__ == "__main__":
    main()
