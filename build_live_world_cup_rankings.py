from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from datetime import datetime

from build_fifa_annual_rankings import normalized_team
from predict import DATA_DIR, OUTPUT_DIR, canonical_team, schedule
from predict_fifa_profile import FifaRanking, FIFA_RANKING_CSV, RANKING_YEAR


RESULTS_CSV = DATA_DIR / "world_cup_2026_results.csv"
LIVE_RANKINGS_CSV = OUTPUT_DIR / "world_cup_2026_live_rankings.csv"
K_FACTOR = 35.0
HOME_ADVANTAGE_POINTS = 0.0
GOAL_DIFF_CAP = 4


@dataclass(frozen=True)
class Result:
    date_bjt: str
    time_bjt: str
    group: str
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int


def tournament_teams() -> set[str]:
    teams = {canonical_team(team) for match in schedule() for team in (match.team_a, match.team_b)}
    with RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            teams.add(canonical_team(row["team_a"]))
            teams.add(canonical_team(row["team_b"]))
    if len(teams) != 48:
        raise RuntimeError(f"expected 48 qualified teams, got {len(teams)}")
    return teams


def load_base_rankings(teams: set[str]) -> dict[str, FifaRanking]:
    rankings: dict[str, FifaRanking] = {}
    with FIFA_RANKING_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["year"]) != RANKING_YEAR:
                continue
            team = normalized_team(row["team"])
            if team not in teams:
                continue
            rankings[team] = FifaRanking(
                rank=int(row["rank"]),
                points=float(row["total_points"]),
                snapshot_date=row["snapshot_date"],
            )
    missing = sorted(teams - set(rankings))
    if missing:
        raise RuntimeError(f"missing base FIFA ranking for: {', '.join(missing)}")
    return rankings


def load_results() -> list[Result]:
    results: list[Result] = []
    with RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            results.append(
                Result(
                    date_bjt=row["date_bjt"],
                    time_bjt=row["time_bjt"],
                    group=row["group"],
                    team_a=canonical_team(row["team_a"]),
                    team_b=canonical_team(row["team_b"]),
                    goals_a=int(row["goals_a"]),
                    goals_b=int(row["goals_b"]),
                )
            )
    return sorted(results, key=lambda item: datetime.strptime(f"{item.date_bjt} {item.time_bjt}", "%Y-%m-%d %H:%M"))


def actual_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def expected_score(points_for: float, points_against: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(points_for - points_against + HOME_ADVANTAGE_POINTS) / 600.0))


def goal_diff_multiplier(goals_for: int, goals_against: int) -> float:
    diff = min(GOAL_DIFF_CAP, abs(goals_for - goals_against))
    if diff <= 1:
        return 1.0
    return 1.0 + (diff - 1) * 0.35


def apply_result(points: dict[str, float], result: Result) -> None:
    score_a = actual_score(result.goals_a, result.goals_b)
    expected_a = expected_score(points[result.team_a], points[result.team_b])
    multiplier = goal_diff_multiplier(result.goals_a, result.goals_b)
    delta = K_FACTOR * multiplier * (score_a - expected_a)
    points[result.team_a] += delta
    points[result.team_b] -= delta


def build_live_rankings() -> list[dict]:
    teams = tournament_teams()
    base = load_base_rankings(teams)
    points = {team: ranking.points for team, ranking in base.items()}
    played = {team: 0 for team in teams}
    goals_for = {team: 0 for team in teams}
    goals_against = {team: 0 for team in teams}
    for result in load_results():
        apply_result(points, result)
        played[result.team_a] += 1
        played[result.team_b] += 1
        goals_for[result.team_a] += result.goals_a
        goals_for[result.team_b] += result.goals_b
        goals_against[result.team_a] += result.goals_b
        goals_against[result.team_b] += result.goals_a

    rows = []
    ordered = sorted(teams, key=lambda team: points[team], reverse=True)
    for rank, team in enumerate(ordered, start=1):
        rows.append(
            {
                "live_rank": rank,
                "team": team,
                "live_points": f"{points[team]:.6f}",
                "base_fifa_rank": base[team].rank,
                "base_fifa_points": f"{base[team].points:.6f}",
                "points_delta": f"{points[team] - base[team].points:.6f}",
                "played": played[team],
                "goals_for": goals_for[team],
                "goals_against": goals_against[team],
                "goal_diff": goals_for[team] - goals_against[team],
                "snapshot_date": base[team].snapshot_date,
                "source": "FIFA 2026 latest filtered to qualified teams + 2026 World Cup results Elo-style update",
            }
        )
    return rows


def write_rankings(rows: list[dict]) -> None:
    fields = [
        "live_rank",
        "team",
        "live_points",
        "base_fifa_rank",
        "base_fifa_points",
        "points_delta",
        "played",
        "goals_for",
        "goals_against",
        "goal_diff",
        "snapshot_date",
        "source",
    ]
    with LIVE_RANKINGS_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    rows = build_live_rankings()
    write_rankings(rows)
    print(f"CSV: {LIVE_RANKINGS_CSV}")
    for row in rows[:12]:
        print(
            f"{row['live_rank']}. {row['team']} {float(row['live_points']):.1f} "
            f"({float(row['points_delta']):+.1f})"
        )


if __name__ == "__main__":
    main()
