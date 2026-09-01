from __future__ import annotations

import csv
import json
import math
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from predict import RESULTS_CSV, canonical_team, download_results, parse_result_date
from profiles import OUTPUT_DIR


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RANKING_CSV = DATA_DIR / "fifa_rankings_history_datofutbol.csv"
GEO_CACHE = DATA_DIR / "open_meteo_geocoding_cache.json"
WEATHER_CACHE = DATA_DIR / "open_meteo_weather_cache.json"
MATCH_CSV = OUTPUT_DIR / "world_cup_heat_latitude_matches.csv"
SUMMARY_MD = OUTPUT_DIR / "world_cup_heat_latitude_summary.md"

WORLD_CUPS = {
    1994: (date(1994, 6, 17), date(1994, 7, 17)),
    1998: (date(1998, 6, 10), date(1998, 7, 12)),
    2002: (date(2002, 5, 31), date(2002, 6, 30)),
    2006: (date(2006, 6, 9), date(2006, 7, 9)),
    2010: (date(2010, 6, 11), date(2010, 7, 11)),
    2014: (date(2014, 6, 12), date(2014, 7, 13)),
    2018: (date(2018, 6, 14), date(2018, 7, 15)),
    2022: (date(2022, 11, 20), date(2022, 12, 18)),
}

TEAM_ALIASES = {
    "Côte d'Ivoire": "Ivory Coast",
    "Czech Republic": "Czechia",
    "IR Iran": "Iran",
    "Korea DPR": "North Korea",
    "Korea Republic": "South Korea",
    "Serbia and Montenegro": "Serbia",
    "USA": "United States",
    "Yugoslavia": "Serbia",
}

RANKING_ALIASES = {
    "China PR": "China",
    "Czech Republic": "Czechia",
    "Korea DPR": "North Korea",
    "Korea Republic": "South Korea",
    "Serbia and Montenegro": "Serbia",
    "Yugoslavia": "Serbia",
}

TEAM_GEO_NAME = {
    "England": "United Kingdom",
    "Iran": "Iran",
    "Ivory Coast": "Ivory Coast",
    "North Korea": "North Korea",
    "Northern Ireland": "United Kingdom",
    "Republic of Ireland": "Ireland",
    "Scotland": "United Kingdom",
    "Serbia": "Serbia",
    "South Korea": "South Korea",
    "United States": "United States",
    "Wales": "United Kingdom",
}

CITY_COUNTRY_HINT = {
    "United States": "US",
    "France": "FR",
    "Japan": "JP",
    "South Korea": "KR",
    "Germany": "DE",
    "South Africa": "ZA",
    "Brazil": "BR",
    "Russia": "RU",
    "Qatar": "QA",
}

CITY_GEO_NAME = {
    "Brasília": "Brasilia",
    "Cuiabá": "Cuiaba",
    "Saint-Denis": "Saint-Denis",
    "Saint-Étienne": "Saint-Etienne",
    "São Paulo": "Sao Paulo",
    "Washington, D.C.": "Washington",
}

HIGH_LATITUDE_DEGREES = 45.0
HOT_MAX_C = 28.0
RANK_SCALE = 18.0


@dataclass(frozen=True)
class Match:
    year: int
    match_date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    city: str
    country: str
    round_index: int


def normalize_team(name: str) -> str:
    return canonical_team(TEAM_ALIASES.get(name.strip(), name.strip()))


def ranking_team(name: str) -> str:
    return normalize_team(RANKING_ALIASES.get(name.strip(), name.strip()))


def parse_ranking_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def request_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "codex-world-cup-heat-study"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def geocode(name: str, *, country_code: str | None, cache: dict) -> dict:
    key = f"{name}|{country_code or ''}"
    if key in cache:
        return cache[key]

    params = {
        "name": name,
        "count": "10",
        "language": "en",
        "format": "json",
    }
    if country_code:
        params["countryCode"] = country_code
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(params)
    data = request_json(url)
    results = data.get("results") or []
    if not results:
        raise RuntimeError(f"geocoding failed: {name} {country_code or ''}")
    cache[key] = results[0]
    time.sleep(0.08)
    return cache[key]


def weather_for(city: str, country: str, match_date: date, geo_cache: dict, weather_cache: dict) -> dict:
    country_code = CITY_COUNTRY_HINT.get(country)
    geo = geocode(CITY_GEO_NAME.get(city, city), country_code=country_code, cache=geo_cache)
    key = f"{geo['latitude']:.4f}|{geo['longitude']:.4f}|{match_date.isoformat()}"
    if key not in weather_cache:
        params = {
            "latitude": f"{geo['latitude']}",
            "longitude": f"{geo['longitude']}",
            "start_date": match_date.isoformat(),
            "end_date": match_date.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_mean",
            "timezone": "auto",
        }
        url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
        data = request_json(url)
        daily = data["daily"]
        weather_cache[key] = {
            "max_c": daily["temperature_2m_max"][0],
            "mean_c": daily["temperature_2m_mean"][0],
            "latitude": geo["latitude"],
            "longitude": geo["longitude"],
        }
        time.sleep(0.08)
    return weather_cache[key]


def team_latitude(team: str, geo_cache: dict) -> float:
    geo_name = TEAM_GEO_NAME.get(team, team)
    geo = geocode(geo_name, country_code=None, cache=geo_cache)
    return float(geo["latitude"])


def weather_cache_key(geo: dict, match_date: date) -> str:
    return f"{geo['latitude']:.4f}|{geo['longitude']:.4f}|{match_date.isoformat()}"


def cache_weather_range(
    city: str,
    country: str,
    dates: list[date],
    geo_cache: dict,
    weather_cache: dict,
) -> None:
    country_code = CITY_COUNTRY_HINT.get(country)
    geo = geocode(CITY_GEO_NAME.get(city, city), country_code=country_code, cache=geo_cache)
    missing = [match_date for match_date in sorted(set(dates)) if weather_cache_key(geo, match_date) not in weather_cache]
    if not missing:
        return

    params = {
        "latitude": f"{geo['latitude']}",
        "longitude": f"{geo['longitude']}",
        "start_date": min(missing).isoformat(),
        "end_date": max(missing).isoformat(),
        "daily": "temperature_2m_max,temperature_2m_mean",
        "timezone": "auto",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    data = request_json(url)
    daily = data["daily"]
    for day, max_c, mean_c in zip(
        daily["time"],
        daily["temperature_2m_max"],
        daily["temperature_2m_mean"],
        strict=True,
    ):
        match_date = parse_ranking_date(day)
        key = weather_cache_key(geo, match_date)
        weather_cache[key] = {
            "max_c": max_c,
            "mean_c": mean_c,
            "latitude": geo["latitude"],
            "longitude": geo["longitude"],
        }
    time.sleep(0.08)


def prefetch_weather(matches: list[Match], geo_cache: dict, weather_cache: dict) -> None:
    dates_by_city: dict[tuple[str, str], list[date]] = defaultdict(list)
    for match in matches:
        dates_by_city[(match.city, match.country)].append(match.match_date)
    for (city, country), dates in sorted(dates_by_city.items()):
        cache_weather_range(city, country, dates, geo_cache, weather_cache)
        save_json(GEO_CACHE, geo_cache)
        save_json(WEATHER_CACHE, weather_cache)


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
                    "team": ranking_team(row["team"]),
                    "total_points": float(points),
                }
            )

    snapshots: dict[int, dict] = {}
    available_dates = sorted(rows_by_date)
    for year, (start_date, _) in WORLD_CUPS.items():
        snapshot_date = max(ranking_date for ranking_date in available_dates if ranking_date < start_date)
        snapshots[year] = {
            "snapshot_date": snapshot_date,
            "rankings": competition_ranks(rows_by_date[snapshot_date]),
        }
    return snapshots


def load_matches() -> list[Match]:
    download_results()
    raw_matches: dict[int, list[Match]] = defaultdict(list)
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("tournament") != "FIFA World Cup":
                continue
            match_date = parse_result_date(row["date"])
            for year, (start_date, end_date) in WORLD_CUPS.items():
                if start_date <= match_date <= end_date:
                    raw_matches[year].append(
                        Match(
                            year=year,
                            match_date=match_date,
                            home_team=normalize_team(row["home_team"]),
                            away_team=normalize_team(row["away_team"]),
                            home_score=int(row["home_score"]),
                            away_score=int(row["away_score"]),
                            city=row["city"],
                            country=row["country"],
                            round_index=0,
                        )
                    )

    matches: list[Match] = []
    for year in sorted(raw_matches):
        year_matches = sorted(raw_matches[year], key=lambda match: match.match_date)
        matches.extend(
            Match(
                year=match.year,
                match_date=match.match_date,
                home_team=match.home_team,
                away_team=match.away_team,
                home_score=match.home_score,
                away_score=match.away_score,
                city=match.city,
                country=match.country,
                round_index=index,
            )
            for index, match in enumerate(year_matches, start=1)
        )
    return matches


def actual_points(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def expected_points(rank_for: int, rank_against: int) -> float:
    edge = (rank_against - rank_for) / RANK_SCALE
    win_prob = 1.0 / (1.0 + math.exp(-edge))
    draw_prob = max(0.16, 0.30 - min(abs(rank_for - rank_against), 60) * 0.002)
    decisive = 1.0 - draw_prob
    return decisive * win_prob + draw_prob * 0.5


def row_for_team(
    match: Match,
    team: str,
    opponent: str,
    goals_for: int,
    goals_against: int,
    rankings: dict,
    geo_cache: dict,
    weather_cache: dict,
) -> dict:
    if team not in rankings or opponent not in rankings:
        raise RuntimeError(f"missing ranking: {match.year} {team} {opponent}")
    weather = weather_for(match.city, match.country, match.match_date, geo_cache, weather_cache)
    latitude = team_latitude(team, geo_cache)
    opponent_latitude = team_latitude(opponent, geo_cache)
    rank = rankings[team]["rank"]
    opponent_rank = rankings[opponent]["rank"]
    expected = expected_points(rank, opponent_rank)
    actual = actual_points(goals_for, goals_against)
    return {
        "year": match.year,
        "date": match.match_date.isoformat(),
        "stage": "group" if match.round_index <= 48 else "knockout",
        "city": match.city,
        "country": match.country,
        "team": team,
        "opponent": opponent,
        "score_for": goals_for,
        "score_against": goals_against,
        "fifa_rank": rank,
        "opponent_fifa_rank": opponent_rank,
        "expected_points": round(expected, 4),
        "actual_points": actual,
        "points_delta": round(actual - expected, 4),
        "team_latitude": round(latitude, 4),
        "opponent_latitude": round(opponent_latitude, 4),
        "latitude_abs": round(abs(latitude), 4),
        "opponent_latitude_abs": round(abs(opponent_latitude), 4),
        "high_latitude_team": abs(latitude) >= HIGH_LATITUDE_DEGREES,
        "temperature_max_c": weather["max_c"],
        "temperature_mean_c": weather["mean_c"],
        "hot_match": weather["max_c"] >= HOT_MAX_C,
    }


def avg(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def bucket_line(label: str, rows: list[dict]) -> str:
    if not rows:
        return f"| {label} | 0 | - | - | - |"
    deltas = [float(row["points_delta"]) for row in rows]
    under = sum(1 for value in deltas if value < 0)
    return f"| {label} | {len(rows)} | {avg(deltas):+.3f} | {under / len(rows):.1%} | {avg([float(row['temperature_max_c']) for row in rows]):.1f} |"


def write_outputs(rows: list[dict]) -> None:
    fields = [
        "year",
        "date",
        "stage",
        "city",
        "country",
        "team",
        "opponent",
        "score_for",
        "score_against",
        "fifa_rank",
        "opponent_fifa_rank",
        "expected_points",
        "actual_points",
        "points_delta",
        "team_latitude",
        "opponent_latitude",
        "latitude_abs",
        "opponent_latitude_abs",
        "high_latitude_team",
        "temperature_max_c",
        "temperature_mean_c",
        "hot_match",
    ]
    with MATCH_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    high_hot = [r for r in rows if r["high_latitude_team"] and r["hot_match"]]
    high_cool = [r for r in rows if r["high_latitude_team"] and not r["hot_match"]]
    low_hot = [r for r in rows if not r["high_latitude_team"] and r["hot_match"]]
    low_cool = [r for r in rows if not r["high_latitude_team"] and not r["hot_match"]]
    group_high_hot = [r for r in high_hot if r["stage"] == "group"]
    group_high_cool = [r for r in high_cool if r["stage"] == "group"]

    markdown = [
        "# 世界杯高温与高纬球队历史验证",
        "",
        "- 样本：1994-2022 世界杯正赛，每场拆成两条球队记录。",
        "- 预期分：用赛前 FIFA 排名估算，胜=1，平=0.5，负=0。",
        f"- 高纬球队：国家代表纬度绝对值 `>= {HIGH_LATITUDE_DEGREES:.0f}°`。",
        f"- 热天：比赛城市当日最高温 `>= {HOT_MAX_C:.0f}°C`，天气来自 Open-Meteo 历史接口。",
        "- 结论只能作为模型修正信号，不是严格因果证明。",
        "",
        "| 分组 | 球队记录数 | 实际分-预期分 | 低于预期比例 | 平均最高温 |",
        "|---|---:|---:|---:|---:|",
        bucket_line("高纬球队 + 热天", high_hot),
        bucket_line("高纬球队 + 非热天", high_cool),
        bucket_line("非高纬球队 + 热天", low_hot),
        bucket_line("非高纬球队 + 非热天", low_cool),
        bucket_line("小组赛：高纬球队 + 热天", group_high_hot),
        bucket_line("小组赛：高纬球队 + 非热天", group_high_cool),
        "",
        "## 热天高纬球队低于预期场次",
        "",
        "| 年份 | 日期 | 城市 | 球队 | 对手 | 比分 | FIFA排名 | 预期分 | 实际分 | 差值 | 最高温 |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    worst = sorted(high_hot, key=lambda row: float(row["points_delta"]))[:20]
    for row in worst:
        markdown.append(
            f"| {row['year']} | {row['date']} | {row['city']} | {row['team']} | {row['opponent']} | "
            f"{row['score_for']}-{row['score_against']} | {row['fifa_rank']}-{row['opponent_fifa_rank']} | "
            f"{float(row['expected_points']):.2f} | {float(row['actual_points']):.1f} | "
            f"{float(row['points_delta']):+.2f} | {float(row['temperature_max_c']):.1f} |"
        )
    SUMMARY_MD.write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    geo_cache = load_json(GEO_CACHE)
    weather_cache = load_json(WEATHER_CACHE)
    snapshots = load_ranking_snapshots()
    matches = load_matches()
    prefetch_weather(matches, geo_cache, weather_cache)
    rows: list[dict] = []

    for match in matches:
        rankings = snapshots[match.year]["rankings"]
        rows.append(
            row_for_team(
                match,
                match.home_team,
                match.away_team,
                match.home_score,
                match.away_score,
                rankings,
                geo_cache,
                weather_cache,
            )
        )
        rows.append(
            row_for_team(
                match,
                match.away_team,
                match.home_team,
                match.away_score,
                match.home_score,
                rankings,
                geo_cache,
                weather_cache,
            )
        )
        save_json(GEO_CACHE, geo_cache)
        save_json(WEATHER_CACHE, weather_cache)

    save_json(GEO_CACHE, geo_cache)
    save_json(WEATHER_CACHE, weather_cache)
    write_outputs(rows)
    print(f"CSV: {MATCH_CSV}")
    print(f"Markdown: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
