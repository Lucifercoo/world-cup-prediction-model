from __future__ import annotations

import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from predict import RESULTS_CSV, canonical_team, download_results, parse_result_date
from profiles import OUTPUT_DIR


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RANKING_CSV = DATA_DIR / "fifa_rankings_history_datofutbol.csv"
MATCH_CSV = OUTPUT_DIR / "world_cup_fifa_ranking_backtest_matches.csv"
SUMMARY_MD = OUTPUT_DIR / "world_cup_fifa_ranking_backtest_summary.md"
DRAW_RANK_GAP = 10
WORLD_CUPS = {
    2010: (date(2010, 6, 11), date(2010, 7, 11)),
    2014: (date(2014, 6, 12), date(2014, 7, 13)),
    2018: (date(2018, 6, 14), date(2018, 7, 15)),
    2022: (date(2022, 11, 20), date(2022, 12, 18)),
}


ALIASES = {
    "Côte d'Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Korea DPR": "North Korea",
    "Korea Republic": "South Korea",
    "USA": "United States",
}


@dataclass(frozen=True)
class WorldCupMatch:
    year: int
    date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    round_index: int


def normalized_team(name: str) -> str:
    return canonical_team(ALIASES.get(name.strip(), name.strip()))


def parse_ranking_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def outcome(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home"
    if home_score < away_score:
        return "away"
    return "draw"


def outcome_label(value: str) -> str:
    return {"home": "主胜", "draw": "平", "away": "客胜"}[value]


def format_outcomes(values: list[str]) -> str:
    if not values:
        raise ValueError("predicted outcomes cannot be empty")
    return "/".join(outcome_label(value) for value in values)


def stage_bucket(match: WorldCupMatch) -> str:
    if match.round_index <= 48:
        return "group"
    return "knockout"


def load_world_cup_matches() -> list[WorldCupMatch]:
    download_results()
    raw_matches: dict[int, list[WorldCupMatch]] = defaultdict(list)
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("tournament") != "FIFA World Cup":
                continue
            match_date = parse_result_date(row["date"])
            for year, (start_date, end_date) in WORLD_CUPS.items():
                if start_date <= match_date <= end_date:
                    if not row.get("home_score", "").isdigit() or not row.get("away_score", "").isdigit():
                        raise RuntimeError(f"missing score for World Cup {year}: {row}")
                    raw_matches[year].append(
                        WorldCupMatch(
                            year=year,
                            date=match_date,
                            home_team=canonical_team(row["home_team"]),
                            away_team=canonical_team(row["away_team"]),
                            home_score=int(row["home_score"]),
                            away_score=int(row["away_score"]),
                            round_index=0,
                        )
                    )

    matches: list[WorldCupMatch] = []
    for year in sorted(WORLD_CUPS):
        year_matches = sorted(raw_matches[year], key=lambda match: match.date)
        if len(year_matches) != 64:
            raise RuntimeError(f"expected 64 World Cup {year} matches, got {len(year_matches)}")
        matches.extend(
            WorldCupMatch(
                year=match.year,
                date=match.date,
                home_team=match.home_team,
                away_team=match.away_team,
                home_score=match.home_score,
                away_score=match.away_score,
                round_index=index,
            )
            for index, match in enumerate(year_matches, start=1)
        )
    return matches


def competition_ranks(rows: list[dict]) -> dict[str, dict]:
    sorted_rows = sorted(rows, key=lambda row: (-row["total_points"], row["team"]))
    rankings: dict[str, dict] = {}
    previous_points: float | None = None
    previous_rank: int | None = None
    for index, row in enumerate(sorted_rows, start=1):
        rank = previous_rank if previous_points == row["total_points"] else index
        if rank is None:
            raise RuntimeError("ranking failed")
        rankings[row["team"]] = {
            "rank": rank,
            "total_points": row["total_points"],
        }
        previous_points = row["total_points"]
        previous_rank = rank
    return rankings


def load_ranking_snapshots() -> dict[int, dict]:
    rows_by_date: dict[date, list[dict]] = defaultdict(list)
    with RANKING_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            points = row["total_points"].strip()
            if points in {"", "NA"}:
                continue
            ranking_date = parse_ranking_date(row["date"])
            rows_by_date[ranking_date].append(
                {
                    "team": normalized_team(row["team"]),
                    "total_points": float(points),
                }
            )

    snapshots: dict[int, dict] = {}
    available_dates = sorted(rows_by_date)
    for year, (start_date, _) in WORLD_CUPS.items():
        snapshot_date = max(ranking_date for ranking_date in available_dates if ranking_date < start_date)
        rankings = competition_ranks(rows_by_date[snapshot_date])
        if len(rankings) < 200:
            raise RuntimeError(f"expected at least 200 teams in FIFA ranking {snapshot_date}, got {len(rankings)}")
        snapshots[year] = {
            "snapshot_date": snapshot_date,
            "rankings": rankings,
        }
    return snapshots


def predicted_outcomes(home_rank: int, away_rank: int) -> list[str]:
    if home_rank < away_rank:
        picks = ["home"]
    elif home_rank > away_rank:
        picks = ["away"]
    else:
        picks = ["draw"]
    if abs(home_rank - away_rank) <= DRAW_RANK_GAP and "draw" not in picks:
        picks.append("draw")
    return picks


def average_prediction_count(rows: list[dict]) -> float:
    return sum(int(row["prediction_count"]) for row in rows) / len(rows)


def summary_line(label: str, rows: list[dict]) -> str:
    if not rows:
        return f"| {label} | 0 | 0 | - | - |"
    correct = sum(1 for row in rows if row["correct"])
    return (
        f"| {label} | {len(rows)} | {correct} | "
        f"{correct / len(rows):.1%} | {average_prediction_count(rows):.2f} |"
    )


def write_match_csv(rows: list[dict]) -> None:
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
        "rank_diff_abs",
        "rank_diff_home_minus_away",
        "home_fifa_points",
        "away_fifa_points",
        "ranking_snapshot_date",
    ]
    with MATCH_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict], snapshots: dict[int, dict]) -> None:
    errors = [row for row in rows if not row["correct"]]
    big_gap_errors = sorted(
        [row for row in errors if int(row["rank_diff_abs"]) >= 20],
        key=lambda row: int(row["rank_diff_abs"]),
        reverse=True,
    )

    lines = [
        "# 世界杯 FIFA 排名基线回测",
        "",
        f"- 覆盖年份：{', '.join(str(year) for year in sorted(WORLD_CUPS))}",
        "- 金标准：对应年份世界杯 64 场正赛赛果。",
        "- 排名取法：每届世界杯开赛前最近一期 FIFA 男足排名。",
        f"- 预测规则：排名号更小者胜；排名差 <= {DRAW_RANK_GAP} 时额外加入平局候选。",
        "",
        "## 排名快照",
        "",
        "| 年份 | 开赛日 | 排名日期 | 队伍数 |",
        "|---:|---|---|---:|",
    ]
    for year in sorted(WORLD_CUPS):
        start_date, _ = WORLD_CUPS[year]
        snapshot = snapshots[year]
        lines.append(
            f"| {year} | {start_date.isoformat()} | {snapshot['snapshot_date'].isoformat()} | "
            f"{len(snapshot['rankings'])} |"
        )

    lines.extend(
        [
            "",
            "## 命中率",
            "",
            "| 范围 | 场次 | 命中 | 命中率 | 平均候选数 |",
            "|---|---:|---:|---:|---:|",
            summary_line("全部", rows),
        ]
    )
    for year in sorted(WORLD_CUPS):
        lines.append(summary_line(str(year), [row for row in rows if int(row["year"]) == year]))

    lines.extend(
        [
            summary_line("小组赛", [row for row in rows if row["stage"] == "group"]),
            summary_line("淘汰赛", [row for row in rows if row["stage"] == "knockout"]),
            summary_line("单候选场次", [row for row in rows if int(row["prediction_count"]) == 1]),
            summary_line("双候选场次", [row for row in rows if int(row["prediction_count"]) == 2]),
            "",
            "## 大排名差但预测错",
            "",
            "| 年份 | 日期 | 比赛 | 实际 | 预测 | 排名差 |",
            "|---:|---|---|---|---|---:|",
        ]
    )
    for row in big_gap_errors[:20]:
        lines.append(
            "| {year} | {date} | {home} {score} {away} | {actual} | {predicted} | {gap} |".format(
                year=row["year"],
                date=row["date"],
                home=row["home_team"],
                score=row["score"],
                away=row["away_team"],
                actual=row["actual_outcome"],
                predicted=row["predicted_outcomes"],
                gap=row["rank_diff_abs"],
            )
        )

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_rows(matches: list[WorldCupMatch], snapshots: dict[int, dict]) -> list[dict]:
    rows: list[dict] = []
    for match in matches:
        snapshot = snapshots[match.year]
        rankings = snapshot["rankings"]
        missing = [team for team in (match.home_team, match.away_team) if team not in rankings]
        if missing:
            raise RuntimeError(
                f"missing FIFA ranking for World Cup {match.year} snapshot "
                f"{snapshot['snapshot_date']}: {', '.join(missing)}"
            )
        home = rankings[match.home_team]
        away = rankings[match.away_team]
        actual = outcome(match.home_score, match.away_score)
        predicted = predicted_outcomes(home["rank"], away["rank"])
        rows.append(
            {
                "year": match.year,
                "date": match.date.isoformat(),
                "stage": stage_bucket(match),
                "home_team": match.home_team,
                "away_team": match.away_team,
                "score": f"{match.home_score}-{match.away_score}",
                "actual_outcome": outcome_label(actual),
                "predicted_outcomes": format_outcomes(predicted),
                "prediction_count": len(predicted),
                "correct": actual in predicted,
                "home_fifa_rank": home["rank"],
                "away_fifa_rank": away["rank"],
                "rank_diff_abs": abs(home["rank"] - away["rank"]),
                "rank_diff_home_minus_away": home["rank"] - away["rank"],
                "home_fifa_points": f"{home['total_points']:.2f}",
                "away_fifa_points": f"{away['total_points']:.2f}",
                "ranking_snapshot_date": snapshot["snapshot_date"].isoformat(),
            }
        )
    return rows


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    matches = load_world_cup_matches()
    snapshots = load_ranking_snapshots()
    rows = build_rows(matches, snapshots)
    write_match_csv(rows)
    write_summary(rows, snapshots)

    correct = sum(1 for row in rows if row["correct"])
    print(f"Matches: {MATCH_CSV}")
    print(f"Summary: {SUMMARY_MD}")
    print(f"Accuracy: {correct}/{len(rows)} = {correct / len(rows):.1%}")
    print(f"Average candidates: {average_prediction_count(rows):.2f}")


if __name__ == "__main__":
    main()
