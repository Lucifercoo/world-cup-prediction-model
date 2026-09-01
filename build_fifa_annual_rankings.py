from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from predict import canonical_team


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
SOURCE_CSV = DATA_DIR / "fifa_rankings_history_datofutbol.csv"
OFFICIAL_SNAPSHOTS_CSV = DATA_DIR / "fifa_rankings_official_snapshots.csv"
ANNUAL_CSV = DATA_DIR / "fifa_rankings_annual_start.csv"
SUMMARY_MD = OUTPUT_DIR / "fifa_rankings_annual_start_summary.md"
SOURCE_NAME = "Dato-Futbol/fifa-ranking"
SOURCE_URL = "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/refs/heads/master/ranking_fifa_historical.csv"
OFFICIAL_SOURCE_NAME = "FIFA official API"
OFFICIAL_PAGE_URL = "https://inside.fifa.com/fifa-world-ranking/men"
REQUIRED_OFFICIAL_YEARS = {2025, 2026}


ALIASES = {
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Brunei Darussalam": "Brunei",
    "Cabo Verde": "Cape Verde",
    "China PR": "China",
    "Chinese Taipei": "Taiwan",
    "Congo DR": "DR Congo",
    "Curaçao": "Curaçao",
    "Curacao": "Curaçao",
    "Czech Republic": "Czechia",
    "Czechia": "Czechia",
    "Côte d'Ivoire": "Ivory Coast",
    "DPR Korea": "North Korea",
    "Hong Kong, China": "Hong Kong",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Kyrgyz Republic": "Kyrgyzstan",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "The Gambia": "Gambia",
    "Türkiye": "Turkey",
    "Turkey": "Turkey",
    "US Virgin Islands": "United States Virgin Islands",
    "USA": "United States",
}


def normalized_team(name: str) -> str:
    return canonical_team(ALIASES.get(name.strip(), name.strip()))


def parse_date(value: str):
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_source_rows() -> list[dict]:
    rows: list[dict] = []
    with SOURCE_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            points = row["total_points"].strip()
            if points in {"", "NA"}:
                continue
            rows.append(
                {
                    "team": normalized_team(row["team"]),
                    "team_short": row["team_short"].strip(),
                    "total_points": float(points),
                    "date": parse_date(row["date"]),
                }
            )
    if not rows:
        raise RuntimeError(f"no usable rows found in {SOURCE_CSV}")
    return rows


def annual_snapshot_dates(rows: list[dict]) -> dict[int, object]:
    dates_by_year: dict[int, set] = defaultdict(set)
    for row in rows:
        dates_by_year[row["date"].year].add(row["date"])
    return {year: min(dates) for year, dates in dates_by_year.items()}


def assign_competition_ranks(rows: list[dict]) -> list[dict]:
    sorted_rows = sorted(rows, key=lambda row: (-row["total_points"], row["team"]))
    ranked: list[dict] = []
    previous_points: float | None = None
    previous_rank: int | None = None
    for index, row in enumerate(sorted_rows, start=1):
        if previous_points is not None and row["total_points"] == previous_points:
            rank = previous_rank
        else:
            rank = index
        if rank is None:
            raise RuntimeError("rank assignment failed")
        ranked.append({**row, "rank": rank})
        previous_points = row["total_points"]
        previous_rank = rank
    return ranked


def build_annual_rows(source_rows: list[dict]) -> list[dict]:
    snapshot_dates = annual_snapshot_dates(source_rows)
    rows_by_date: dict[object, list[dict]] = defaultdict(list)
    for row in source_rows:
        rows_by_date[row["date"]].append(row)

    annual_rows: list[dict] = []
    for year in sorted(snapshot_dates):
        snapshot_date = snapshot_dates[year]
        for row in assign_competition_ranks(rows_by_date[snapshot_date]):
            annual_rows.append(
                {
                    "year": year,
                    "snapshot_date": snapshot_date.isoformat(),
                    "rank": row["rank"],
                    "team": row["team"],
                    "team_short": row["team_short"],
                    "total_points": f"{row['total_points']:.2f}",
                    "source": SOURCE_NAME,
                }
            )
    return annual_rows


def load_official_snapshot_rows() -> list[dict]:
    if not OFFICIAL_SNAPSHOTS_CSV.exists():
        raise RuntimeError(
            f"missing restricted input {OFFICIAL_SNAPSHOTS_CSV}; "
            "see docs/DATA_FETCH.csv and docs/DATA_SOURCES.md"
        )

    rows: list[dict] = []
    with OFFICIAL_SNAPSHOTS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rank = row["rank"].strip()
            if rank == "":
                continue
            rows.append(
                {
                    "year": int(row["year"]),
                    "snapshot_date": row["snapshot_date"],
                    "rank": int(rank),
                    "team": normalized_team(row["team"]),
                    "team_short": row["team_short"].strip(),
                    "total_points": f"{float(row['total_points']):.2f}",
                    "source": f"{OFFICIAL_SOURCE_NAME} ({row['snapshot_policy']})",
                }
            )

    years = {int(row["year"]) for row in rows}
    missing_years = REQUIRED_OFFICIAL_YEARS - years
    if missing_years:
        missing = ", ".join(str(year) for year in sorted(missing_years))
        raise RuntimeError(f"official FIFA snapshots missing required years: {missing}")
    return rows


def combine_annual_rows(source_rows: list[dict], official_rows: list[dict]) -> list[dict]:
    official_years = {int(row["year"]) for row in official_rows}
    combined = [row for row in source_rows if int(row["year"]) not in official_years]
    combined.extend(official_rows)
    return sorted(combined, key=lambda row: (int(row["year"]), int(row["rank"])))


def write_annual_csv(rows: list[dict]) -> None:
    fields = ["year", "snapshot_date", "rank", "team", "team_short", "total_points", "source"]
    with ANNUAL_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict]) -> None:
    years = sorted({int(row["year"]) for row in rows})
    rows_by_year: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_year[int(row["year"])].append(row)

    lines = [
        "# FIFA 年初排名基础数据",
        "",
        f"- 历史来源：{SOURCE_NAME}",
        f"- 历史原始数据：{SOURCE_URL}",
        f"- 官方补充来源：{OFFICIAL_PAGE_URL}",
        f"- 年份范围：{years[0]} 到 {years[-1]}",
        "- 取数规则：1992-2025 取该年最早一期 FIFA 男足排名；2026 按当前预测需求取最新一期。",
        "- rank 说明：原始文件只有积分，本文件按同一期 total_points 降序重新计算名次；同分使用并列名次。",
        "- 官方补充说明：FIFA 官方返回 `rank=null` 的球队不写入年度排名。",
        "",
        "## 每年快照",
        "",
        "| 年份 | 快照日期 | 队伍数 | 第一名 | 积分 |",
        "|---:|---|---:|---|---:|",
    ]
    for year in years:
        year_rows = sorted(rows_by_year[year], key=lambda row: int(row["rank"]))
        leader = year_rows[0]
        lines.append(
            f"| {year} | {leader['snapshot_date']} | {len(year_rows)} | {leader['team']} | {float(leader['total_points']):.2f} |"
        )

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    source_rows = load_source_rows()
    annual_rows = combine_annual_rows(build_annual_rows(source_rows), load_official_snapshot_rows())
    write_annual_csv(annual_rows)
    write_summary(annual_rows)
    years = sorted({int(row["year"]) for row in annual_rows})
    print(f"Wrote {ANNUAL_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Years: {years[0]}-{years[-1]}")
    print(f"Rows: {len(annual_rows)}")


if __name__ == "__main__":
    main()
