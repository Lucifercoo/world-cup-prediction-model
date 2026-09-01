from __future__ import annotations

import csv
import math
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
RESULTS_CSV = DATA_DIR / "international_results.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "group_score_predictions.csv"
PREDICTIONS_MD = OUTPUT_DIR / "group_score_predictions.md"

AS_OF = date(2026, 6, 12)
ROLLING_YEARS = 10
TRAINING_START = date(AS_OF.year - ROLLING_YEARS, AS_OF.month, AS_OF.day)
HALF_LIFE_DAYS = 900.0
PSEUDO_MATCHES = 8.0
BASE_ELO = 1500.0
MAX_GOALS = 8
DIXON_COLES_RHO = -0.08


ALIASES = {
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "Curaçao": "Curaçao",
    "Czech Republic": "Czechia",
    "Czechia": "Czechia",
    "DR Congo": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Ivory Coast": "Ivory Coast",
    "South Korea": "South Korea",
    "Türkiye": "Turkey",
    "Turkey": "Turkey",
    "United States": "United States",
}


HOST_COUNTRY_BY_VENUE = {
    "Mexico City": "Mexico",
    "Zapopan": "Mexico",
    "Guadalupe": "Mexico",
    "Toronto": "Canada",
    "Vancouver": "Canada",
}


@dataclass(frozen=True)
class Match:
    group: str
    day_et: date
    time_et: str
    team_a: str
    team_b: str
    venue: str


@dataclass(frozen=True)
class TeamStats:
    matches: float
    gf_per_match: float
    ga_per_match: float
    recent_gd: float
    elo: float


def canonical_team(name: str) -> str:
    return ALIASES.get(name.strip(), name.strip())


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_results() -> None:
    ensure_dirs()
    if RESULTS_CSV.exists():
        return
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "codex-wc2026-score-model"})
    with urllib.request.urlopen(req, timeout=30) as response:
        RESULTS_CSV.write_bytes(response.read())


def parse_result_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_results(start_date: date) -> list[dict]:
    download_results()
    rows: list[dict] = []
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            match_date = parse_result_date(row["date"])
            if match_date < start_date or match_date > AS_OF:
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

    known_wc_results = [
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
    existing = {
        (r["date"], r["home_team"], r["away_team"], r["home_score"], r["away_score"])
        for r in rows
    }
    for row in known_wc_results:
        key = (row["date"], row["home_team"], row["away_team"], row["home_score"], row["away_score"])
        if key not in existing:
            rows.append(row)

    rows.sort(key=lambda r: r["date"])
    return rows


def schedule() -> list[Match]:
    return [
        Match("A", date(2026, 6, 11), "12:00", "Mexico", "South Africa", "Mexico City"),
        Match("A", date(2026, 6, 11), "18:00", "South Korea", "Czechia", "Guadalupe"),
        Match("B", date(2026, 6, 12), "15:00", "Canada", "Bosnia and Herzegovina", "Toronto"),
        Match("D", date(2026, 6, 12), "21:00", "United States", "Paraguay", "Inglewood"),
        Match("B", date(2026, 6, 13), "15:00", "Qatar", "Switzerland", "Santa Clara"),
        Match("C", date(2026, 6, 13), "18:00", "Brazil", "Morocco", "East Rutherford"),
        Match("C", date(2026, 6, 13), "21:00", "Haiti", "Scotland", "Foxborough"),
        Match("D", date(2026, 6, 14), "00:00", "Australia", "Turkey", "Vancouver"),
        Match("E", date(2026, 6, 14), "13:00", "Germany", "Curaçao", "Houston"),
        Match("F", date(2026, 6, 14), "16:00", "Netherlands", "Japan", "Arlington"),
        Match("E", date(2026, 6, 14), "19:00", "Ivory Coast", "Ecuador", "Philadelphia"),
        Match("F", date(2026, 6, 14), "22:00", "Sweden", "Tunisia", "Guadalupe"),
        Match("H", date(2026, 6, 15), "13:00", "Spain", "Cape Verde", "Atlanta"),
        Match("G", date(2026, 6, 15), "18:00", "Belgium", "Egypt", "Seattle"),
        Match("H", date(2026, 6, 15), "18:00", "Saudi Arabia", "Uruguay", "Miami Gardens"),
        Match("G", date(2026, 6, 16), "00:00", "Iran", "New Zealand", "Inglewood"),
        Match("I", date(2026, 6, 16), "15:00", "France", "Senegal", "East Rutherford"),
        Match("I", date(2026, 6, 16), "18:00", "Iraq", "Norway", "Foxborough"),
        Match("J", date(2026, 6, 16), "21:00", "Argentina", "Algeria", "Kansas City"),
        Match("J", date(2026, 6, 17), "00:00", "Austria", "Jordan", "Santa Clara"),
        Match("K", date(2026, 6, 17), "13:00", "Portugal", "DR Congo", "Houston"),
        Match("L", date(2026, 6, 17), "16:00", "England", "Croatia", "Arlington"),
        Match("L", date(2026, 6, 17), "19:00", "Ghana", "Panama", "Toronto"),
        Match("K", date(2026, 6, 17), "22:00", "Uzbekistan", "Colombia", "Mexico City"),
        Match("A", date(2026, 6, 18), "12:00", "Czechia", "South Africa", "Atlanta"),
        Match("B", date(2026, 6, 18), "15:00", "Switzerland", "Bosnia and Herzegovina", "Inglewood"),
        Match("B", date(2026, 6, 18), "18:00", "Canada", "Qatar", "Vancouver"),
        Match("A", date(2026, 6, 18), "23:00", "Mexico", "South Korea", "Zapopan"),
        Match("D", date(2026, 6, 19), "15:00", "United States", "Australia", "Seattle"),
        Match("C", date(2026, 6, 19), "18:00", "Scotland", "Morocco", "Foxborough"),
        Match("C", date(2026, 6, 19), "21:00", "Brazil", "Haiti", "Philadelphia"),
        Match("D", date(2026, 6, 20), "00:00", "Turkey", "Paraguay", "Santa Clara"),
        Match("F", date(2026, 6, 20), "13:00", "Netherlands", "Sweden", "Houston"),
        Match("E", date(2026, 6, 20), "16:00", "Germany", "Ivory Coast", "Toronto"),
        Match("E", date(2026, 6, 20), "20:00", "Ecuador", "Curaçao", "Kansas City"),
        Match("F", date(2026, 6, 21), "00:00", "Tunisia", "Japan", "Guadalupe"),
        Match("H", date(2026, 6, 21), "12:00", "Spain", "Saudi Arabia", "Atlanta"),
        Match("G", date(2026, 6, 21), "15:00", "Belgium", "Iran", "Inglewood"),
        Match("H", date(2026, 6, 21), "18:00", "Uruguay", "Cape Verde", "Miami Gardens"),
        Match("G", date(2026, 6, 21), "21:00", "New Zealand", "Egypt", "Vancouver"),
        Match("J", date(2026, 6, 22), "13:00", "Argentina", "Austria", "Arlington"),
        Match("I", date(2026, 6, 22), "17:00", "France", "Iraq", "Philadelphia"),
        Match("I", date(2026, 6, 22), "20:00", "Norway", "Senegal", "East Rutherford"),
        Match("J", date(2026, 6, 22), "23:00", "Jordan", "Algeria", "Santa Clara"),
        Match("K", date(2026, 6, 23), "13:00", "Portugal", "Uzbekistan", "Houston"),
        Match("L", date(2026, 6, 23), "16:00", "England", "Ghana", "Foxborough"),
        Match("L", date(2026, 6, 23), "19:00", "Panama", "Croatia", "Toronto"),
        Match("K", date(2026, 6, 23), "22:00", "Colombia", "DR Congo", "Zapopan"),
        Match("B", date(2026, 6, 24), "15:00", "Switzerland", "Canada", "Vancouver"),
        Match("B", date(2026, 6, 24), "15:00", "Bosnia and Herzegovina", "Qatar", "Seattle"),
        Match("C", date(2026, 6, 24), "18:00", "Scotland", "Brazil", "Miami Gardens"),
        Match("C", date(2026, 6, 24), "18:00", "Morocco", "Haiti", "Atlanta"),
        Match("A", date(2026, 6, 24), "21:00", "Czechia", "Mexico", "Mexico City"),
        Match("A", date(2026, 6, 24), "21:00", "South Africa", "South Korea", "Guadalupe"),
        Match("E", date(2026, 6, 25), "16:00", "Ecuador", "Germany", "East Rutherford"),
        Match("E", date(2026, 6, 25), "16:00", "Curaçao", "Ivory Coast", "Philadelphia"),
        Match("F", date(2026, 6, 25), "19:00", "Japan", "Sweden", "Arlington"),
        Match("F", date(2026, 6, 25), "19:00", "Tunisia", "Netherlands", "Kansas City"),
        Match("D", date(2026, 6, 25), "22:00", "Turkey", "United States", "Inglewood"),
        Match("D", date(2026, 6, 25), "22:00", "Paraguay", "Australia", "Santa Clara"),
        Match("I", date(2026, 6, 26), "15:00", "Norway", "France", "Foxborough"),
        Match("I", date(2026, 6, 26), "15:00", "Senegal", "Iraq", "Toronto"),
        Match("H", date(2026, 6, 26), "20:00", "Cape Verde", "Saudi Arabia", "Houston"),
        Match("H", date(2026, 6, 26), "20:00", "Uruguay", "Spain", "Zapopan"),
        Match("G", date(2026, 6, 26), "23:00", "Egypt", "Iran", "Seattle"),
        Match("G", date(2026, 6, 26), "23:00", "New Zealand", "Belgium", "Vancouver"),
        Match("L", date(2026, 6, 27), "17:00", "Panama", "England", "East Rutherford"),
        Match("L", date(2026, 6, 27), "17:00", "Croatia", "Ghana", "Philadelphia"),
        Match("K", date(2026, 6, 27), "19:30", "Colombia", "Portugal", "Miami Gardens"),
        Match("K", date(2026, 6, 27), "19:30", "DR Congo", "Uzbekistan", "Atlanta"),
        Match("J", date(2026, 6, 27), "22:00", "Algeria", "Austria", "Kansas City"),
        Match("J", date(2026, 6, 27), "22:00", "Jordan", "Argentina", "Arlington"),
        Match("R32", date(2026, 6, 28), "15:00", "South Africa", "Canada", "Inglewood"),
        Match("R32", date(2026, 6, 29), "13:00", "Brazil", "Japan", "Houston"),
        Match("R32", date(2026, 6, 29), "16:30", "Germany", "Paraguay", "Foxborough"),
        Match("R32", date(2026, 6, 29), "21:00", "Netherlands", "Morocco", "Guadalupe"),
        Match("R32", date(2026, 6, 30), "13:00", "Ivory Coast", "Norway", "Arlington"),
        Match("R32", date(2026, 6, 30), "17:00", "France", "Sweden", "East Rutherford"),
        Match("R32", date(2026, 6, 30), "21:00", "Mexico", "Ecuador", "Mexico City"),
        Match("R32", date(2026, 7, 1), "12:00", "England", "DR Congo", "Atlanta"),
        Match("R32", date(2026, 7, 1), "16:00", "Belgium", "Senegal", "Seattle"),
        Match("R32", date(2026, 7, 1), "20:00", "United States", "Bosnia and Herzegovina", "Santa Clara"),
        Match("R32", date(2026, 7, 2), "15:00", "Spain", "Austria", "Inglewood"),
        Match("R32", date(2026, 7, 2), "19:00", "Portugal", "Croatia", "Toronto"),
        Match("R32", date(2026, 7, 2), "23:00", "Switzerland", "Algeria", "Vancouver"),
        Match("R32", date(2026, 7, 3), "14:00", "Australia", "Egypt", "Arlington"),
        Match("R32", date(2026, 7, 3), "18:00", "Argentina", "Cape Verde", "Miami Gardens"),
        Match("R32", date(2026, 7, 3), "21:30", "Colombia", "Ghana", "Kansas City"),
        Match("R16", date(2026, 7, 4), "13:00", "Canada", "Morocco", "Houston"),
        Match("R16", date(2026, 7, 4), "17:00", "Paraguay", "France", "Philadelphia"),
        Match("R16", date(2026, 7, 5), "16:00", "Brazil", "Norway", "East Rutherford"),
        Match("R16", date(2026, 7, 5), "20:00", "Mexico", "England", "Mexico City"),
        Match("R16", date(2026, 7, 6), "15:00", "Portugal", "Spain", "Arlington"),
        Match("R16", date(2026, 7, 6), "20:00", "United States", "Belgium", "Seattle"),
        Match("R16", date(2026, 7, 7), "12:00", "Argentina", "Egypt", "Atlanta"),
        Match("R16", date(2026, 7, 7), "16:00", "Switzerland", "Colombia", "Vancouver"),
        Match("QF", date(2026, 7, 9), "16:00", "France", "Morocco", "Boston"),
        Match("QF", date(2026, 7, 10), "15:00", "Spain", "Belgium", "Los Angeles"),
        Match("QF", date(2026, 7, 11), "17:00", "Norway", "England", "Miami Gardens"),
        Match("QF", date(2026, 7, 11), "21:00", "Argentina", "Switzerland", "Kansas City"),
        Match("SF", date(2026, 7, 14), "15:00", "France", "Spain", "Arlington"),
        Match("SF", date(2026, 7, 15), "15:00", "England", "Argentina", "Atlanta"),
        Match("3P", date(2026, 7, 18), "17:00", "France", "England", "Miami Gardens"),
        Match("FINAL", date(2026, 7, 19), "15:00", "Spain", "Argentina", "East Rutherford"),
    ]


def result_weight(match_date: date) -> float:
    age_days = max(0, (AS_OF - match_date).days)
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def elo_k(tournament: str) -> float:
    text = tournament.lower()
    if "world cup" in text:
        return 40.0
    if "continental" in text or "euro" in text or "copa" in text or "africa" in text or "asian" in text or "gold cup" in text:
        return 32.0
    if "qualification" in text or "qualifier" in text or "nations league" in text:
        return 28.0
    if "friendly" in text:
        return 16.0
    return 22.0


def expected_elo_score(elo_a: float, elo_b: float, home_advantage_points: float = 0.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a - home_advantage_points) / 400.0))


def actual_score(goals_a: int, goals_b: int) -> float:
    if goals_a > goals_b:
        return 1.0
    if goals_a == goals_b:
        return 0.5
    return 0.0


def goal_diff_multiplier(goals_a: int, goals_b: int) -> float:
    diff = abs(goals_a - goals_b)
    if diff <= 1:
        return 1.0
    return 1.0 + math.log(diff)


def compute_elo(results: list[dict]) -> dict[str, float]:
    ratings: dict[str, float] = defaultdict(lambda: BASE_ELO)
    for row in results:
        home = row["home_team"]
        away = row["away_team"]
        home_adv = 0.0 if row["neutral"] else 55.0
        exp_home = expected_elo_score(ratings[home], ratings[away], home_adv)
        score_home = actual_score(row["home_score"], row["away_score"])
        k = elo_k(row["tournament"]) * goal_diff_multiplier(row["home_score"], row["away_score"])
        delta = k * (score_home - exp_home)
        ratings[home] += delta
        ratings[away] -= delta
    return dict(ratings)


def compute_stats(results: list[dict], ratings: dict[str, float]) -> dict[str, TeamStats]:
    gf: dict[str, float] = defaultdict(float)
    ga: dict[str, float] = defaultdict(float)
    weights: dict[str, float] = defaultdict(float)
    recent: dict[str, list[tuple[date, int]]] = defaultdict(list)
    total_goals = 0.0
    total_team_weights = 0.0

    for row in results:
        w = result_weight(row["date"])
        h = row["home_team"]
        a = row["away_team"]
        hs = row["home_score"]
        away_s = row["away_score"]
        for team, goals_for, goals_against in ((h, hs, away_s), (a, away_s, hs)):
            gf[team] += goals_for * w
            ga[team] += goals_against * w
            weights[team] += w
            recent[team].append((row["date"], goals_for - goals_against))
            total_goals += goals_for * w
            total_team_weights += w

    avg_goals = total_goals / total_team_weights if total_team_weights else 1.25
    stats: dict[str, TeamStats] = {}
    for team in set(weights) | set(ratings):
        w = weights.get(team, 0.0)
        shrunk_gf = (gf.get(team, 0.0) + PSEUDO_MATCHES * avg_goals) / (w + PSEUDO_MATCHES)
        shrunk_ga = (ga.get(team, 0.0) + PSEUDO_MATCHES * avg_goals) / (w + PSEUDO_MATCHES)
        recent_gds = [gd for _, gd in sorted(recent.get(team, []), key=lambda x: x[0])[-8:]]
        recent_gd = sum(recent_gds) / len(recent_gds) if recent_gds else 0.0
        stats[team] = TeamStats(
            matches=w,
            gf_per_match=shrunk_gf,
            ga_per_match=shrunk_ga,
            recent_gd=recent_gd,
            elo=ratings.get(team, BASE_ELO),
        )
    return stats


def dixon_coles_factor(home_goals: int, away_goals: int, lambda_home: float, lambda_away: float) -> float:
    rho = DIXON_COLES_RHO
    if home_goals == 0 and away_goals == 0:
        return 1.0 + lambda_home * lambda_away * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 - lambda_home * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 - lambda_away * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 + rho
    return 1.0


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def score_matrix(lambda_a: float, lambda_b: float) -> list[list[float]]:
    matrix: list[list[float]] = []
    total = 0.0
    for i in range(MAX_GOALS + 1):
        row: list[float] = []
        for j in range(MAX_GOALS + 1):
            p = poisson_pmf(i, lambda_a) * poisson_pmf(j, lambda_b)
            p *= dixon_coles_factor(i, j, lambda_a, lambda_b)
            row.append(p)
            total += p
        matrix.append(row)
    if total <= 0:
        raise RuntimeError("score matrix has zero probability mass")
    return [[p / total for p in row] for row in matrix]


def host_multiplier(team: str, venue: str) -> float:
    host = HOST_COUNTRY_BY_VENUE.get(venue)
    if host is None and team == "United States":
        host = "United States"
    return 1.09 if host == team else 1.0


def expected_goals(match: Match, stats: dict[str, TeamStats]) -> tuple[float, float]:
    a = canonical_team(match.team_a)
    b = canonical_team(match.team_b)
    if a not in stats:
        raise KeyError(f"missing stats for {a}")
    if b not in stats:
        raise KeyError(f"missing stats for {b}")

    avg_goals = sum(s.gf_per_match for s in stats.values()) / len(stats)
    sa = stats[a]
    sb = stats[b]
    attack_a = sa.gf_per_match / avg_goals
    attack_b = sb.gf_per_match / avg_goals
    defense_a = sa.ga_per_match / avg_goals
    defense_b = sb.ga_per_match / avg_goals

    base_a = avg_goals * attack_a * defense_b
    base_b = avg_goals * attack_b * defense_a

    elo_expected_a = expected_elo_score(sa.elo, sb.elo)
    elo_expected_b = 1.0 - elo_expected_a
    elo_scale_a = 1.0 + (elo_expected_a - 0.5) * 0.42
    elo_scale_b = 1.0 + (elo_expected_b - 0.5) * 0.42

    form_scale_a = 1.0 + max(-2.0, min(2.0, sa.recent_gd)) * 0.07
    form_scale_b = 1.0 + max(-2.0, min(2.0, sb.recent_gd)) * 0.07

    lam_a = (0.62 * base_a) + (0.24 * base_a * elo_scale_a) + (0.14 * base_a * form_scale_a)
    lam_b = (0.62 * base_b) + (0.24 * base_b * elo_scale_b) + (0.14 * base_b * form_scale_b)

    lam_a *= host_multiplier(a, match.venue)
    lam_b *= host_multiplier(b, match.venue)

    return max(0.05, min(4.2, lam_a)), max(0.05, min(4.2, lam_b))


def predict_match(match: Match, stats: dict[str, TeamStats]) -> dict:
    lam_a, lam_b = expected_goals(match, stats)
    matrix = score_matrix(lam_a, lam_b)
    cells: list[tuple[int, int, float]] = []
    p_a = p_draw = p_b = 0.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            cells.append((i, j, p))
            if i > j:
                p_a += p
            elif i == j:
                p_draw += p
            else:
                p_b += p

    cells.sort(key=lambda x: x[2], reverse=True)
    most = cells[0]
    outcome_probs = {"A": p_a, "D": p_draw, "B": p_b}
    predicted_outcome = max(outcome_probs, key=outcome_probs.get)
    if predicted_outcome == "A":
        aligned_scores = [cell for cell in cells if cell[0] > cell[1]]
    elif predicted_outcome == "B":
        aligned_scores = [cell for cell in cells if cell[0] < cell[1]]
    else:
        aligned_scores = [cell for cell in cells if cell[0] == cell[1]]
    recommended = aligned_scores[0]
    bjt = datetime.combine(match.day_et, datetime.strptime(match.time_et, "%H:%M").time()) + timedelta(hours=12)
    return {
        "group": match.group,
        "date_bjt": bjt.strftime("%Y-%m-%d"),
        "time_bjt": bjt.strftime("%H:%M"),
        "date_et": match.day_et.isoformat(),
        "time_et": match.time_et,
        "team_a": canonical_team(match.team_a),
        "team_b": canonical_team(match.team_b),
        "venue": match.venue,
        "xg_a": lam_a,
        "xg_b": lam_b,
        "score": f"{most[0]}-{most[1]}",
        "score_probability": most[2],
        "predicted_outcome": predicted_outcome,
        "recommended_score": f"{recommended[0]}-{recommended[1]}",
        "recommended_score_probability": recommended[2],
        "p_a": p_a,
        "p_draw": p_draw,
        "p_b": p_b,
        "top_scores": "; ".join(f"{i}-{j} {p:.1%}" for i, j, p in cells[:3]),
    }


def write_outputs(predictions: list[dict]) -> None:
    fieldnames = [
        "group",
        "date_bjt",
        "time_bjt",
        "date_et",
        "time_et",
        "team_a",
        "team_b",
        "venue",
        "xg_a",
        "xg_b",
        "score",
        "score_probability",
        "predicted_outcome",
        "recommended_score",
        "recommended_score_probability",
        "p_a",
        "p_draw",
        "p_b",
        "top_scores",
    ]
    with PREDICTIONS_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    lines = [
        "# 2026 世界杯剩余小组赛比分预测",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 数据：martj42/international_results；滚动窗口为最近 {ROLLING_YEARS} 年，训练样本从 {TRAINING_START.isoformat()} 到 {AS_OF.isoformat()}；手动加入 2026-06-11 已完赛两场。",
        "- 模型：窗口内 Elo + 窗口内进失球强度 + 近期状态 + Poisson/Dixon-Coles 分数矩阵。",
        "",
        "| 北京时间 | 组 | 比赛 | 赛果预测 | 推荐比分 | 最高单格比分 | 胜/平/负 | xG | 前三比分 |",
        "|---|---:|---|---:|---:|---:|---|---|---|",
    ]
    for row in predictions:
        outcome_label = {
            "A": f"{row['team_a']}胜",
            "D": "平",
            "B": f"{row['team_b']}胜",
        }[row["predicted_outcome"]]
        lines.append(
            "| {date} {time} | {group} | {a} vs {b} | {outcome} | {rec_score} ({rec_p:.1%}) | {score} ({score_p:.1%}) | "
            "{pa:.1%}/{pd:.1%}/{pb:.1%} | {xa:.2f}-{xb:.2f} | {tops} |".format(
                date=row["date_bjt"],
                time=row["time_bjt"],
                group=row["group"],
                a=row["team_a"],
                b=row["team_b"],
                outcome=outcome_label,
                rec_score=row["recommended_score"],
                rec_p=row["recommended_score_probability"],
                score=row["score"],
                score_p=row["score_probability"],
                pa=row["p_a"],
                pd=row["p_draw"],
                pb=row["p_b"],
                xa=row["xg_a"],
                xb=row["xg_b"],
                tops=row["top_scores"],
            )
        )
    PREDICTIONS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    training_results = load_results(TRAINING_START)
    ratings = compute_elo(training_results)
    stats = compute_stats(training_results, ratings)
    predictions = [predict_match(match, stats) for match in schedule()]
    write_outputs(predictions)
    print(f"Wrote {PREDICTIONS_CSV}")
    print(f"Wrote {PREDICTIONS_MD}")
    for row in predictions[:12]:
        print(
            f"{row['date_bjt']} {row['time_bjt']} {row['team_a']} vs {row['team_b']}: "
            f"{row['score']} ({row['score_probability']:.1%})"
        )


if __name__ == "__main__":
    main()
