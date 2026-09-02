from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import urllib.parse
import urllib.request
import zipfile
from datetime import date, datetime
from pathlib import Path

from builders import build_fifa_annual_rankings
from predict import canonical_team, schedule


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SCHEMA_DIR = ROOT / "schemas"
KAGGLE_ARCHIVE_URL = "https://www.kaggle.com/api/v1/datasets/download/cashncarry/fifaworldranking"
KAGGLE_MEMBER = "fifa_ranking-2024-06-20.csv"
KAGGLE_MEMBER_SHA256 = "8fcc1e20a6011b7fb584e0241b4c5056b0507fc27a1af3228f61d409f293c933"
FIFA_RANKING_PAGE_URL = "https://inside.fifa.com/fifa-world-ranking/men"
FIFA_RANKING_ENDPOINT = "https://inside.fifa.com/api/live-world-ranking/get-rankings"
PRE_TOURNAMENT_CUTOFF = date(2026, 6, 11)
HISTORY_CSV = DATA_DIR / "fifa_rankings_history_open.csv"
OFFICIAL_SNAPSHOTS_CSV = DATA_DIR / "fifa_rankings_official_snapshots.csv"
ANNUAL_CSV = DATA_DIR / "fifa_rankings_annual_start.csv"
SQUAD_PROXY_CSV = DATA_DIR / "transfermarkt_world_cup_2026_values.csv"
KEY_PLAYER_SIGNALS_CSV = DATA_DIR / "world_cup_2026_key_player_signals.csv"
PUBLIC_PROXY_MARKER = "PROXY (FIFA points log-linear mapping; NOT Transfermarkt)"

HEADERS = {
    "User-Agent": "world-cup-prediction-model-public-setup/1.0",
    "Accept": "application/json,text/html,*/*",
    "Referer": FIFA_RANKING_PAGE_URL,
}


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def refuse_to_overwrite_user_inputs() -> None:
    if ANNUAL_CSV.exists():
        sources = {row.get("source", "") for row in csv_rows(ANNUAL_CSV)}
        if sources and not any("cashncarry/fifaworldranking" in source for source in sources):
            raise RuntimeError(
                f"refusing to overwrite non-public FIFA ranking input: {ANNUAL_CSV}"
            )
    if SQUAD_PROXY_CSV.exists():
        sources = {row.get("source", "") for row in csv_rows(SQUAD_PROXY_CSV)}
        if sources != {PUBLIC_PROXY_MARKER}:
            raise RuntimeError(
                f"refusing to overwrite user-supplied squad-value input: {SQUAD_PROXY_CSV}"
            )


def build_open_history() -> None:
    archive = request_bytes(KAGGLE_ARCHIVE_URL)
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        if KAGGLE_MEMBER not in zipped.namelist():
            raise RuntimeError(f"Kaggle archive does not contain {KAGGLE_MEMBER}")
        source = zipped.read(KAGGLE_MEMBER)
    actual_hash = hashlib.sha256(source).hexdigest()
    if actual_hash != KAGGLE_MEMBER_SHA256:
        raise RuntimeError(
            f"Kaggle ranking checksum changed: {actual_hash} != {KAGGLE_MEMBER_SHA256}"
        )

    rows: list[dict] = []
    with io.StringIO(source.decode("utf-8-sig")) as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "team": row["country_full"].strip(),
                    "team_short": row["country_abrv"].strip(),
                    "total_points": row["total_points"].strip(),
                    "date": row["rank_date"].strip(),
                    "source": "cashncarry/fifaworldranking CC0",
                }
            )
    if len(rows) < 60_000:
        raise RuntimeError(f"expected at least 60000 historical ranking rows, got {len(rows)}")
    write_csv(HISTORY_CSV, ["team", "team_short", "total_points", "date", "source"], rows)
    print(f"Wrote {HISTORY_CSV} ({len(rows)} rows)")


def available_fifa_dates() -> list[dict]:
    html = request_bytes(FIFA_RANKING_PAGE_URL).decode("utf-8", "replace")
    match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
    if not match:
        raise RuntimeError("FIFA ranking page does not contain __NEXT_DATA__")
    payload = json.loads(match.group(1))
    dates = payload["props"]["pageProps"]["pageData"]["ranking"]["allAvailableDates"]
    if not dates:
        raise RuntimeError("FIFA ranking page returned no snapshot dates")
    return dates


def snapshot_date(row: dict) -> date:
    return datetime.strptime(row["date"], "%Y-%m-%d").date()


def select_official_snapshots(rows: list[dict], cutoff: date) -> list[tuple[int, str, dict]]:
    rows_2025 = [row for row in rows if snapshot_date(row).year == 2025]
    rows_2026 = [
        row for row in rows if snapshot_date(row).year == 2026 and snapshot_date(row) <= cutoff
    ]
    if not rows_2025 or not rows_2026:
        raise RuntimeError(f"FIFA does not provide required snapshots through {cutoff}")
    return [
        (2025, "first", min(rows_2025, key=snapshot_date)),
        (2026, "pre_tournament", max(rows_2026, key=snapshot_date)),
    ]


def fetch_official_ranking(schedule_id: str) -> tuple[list[dict], str]:
    query = urllib.parse.urlencode(
        {
            "mode": "schedule",
            "gender": 1,
            "locale": "en",
            "scheduleId": schedule_id,
            "rankingType": 0,
            "count": 300,
        }
    )
    url = f"{FIFA_RANKING_ENDPOINT}?{query}"
    payload = json.loads(request_bytes(url).decode("utf-8"))
    rankings = payload.get("rankings", [])
    if payload.get("total") != len(rankings) or len(rankings) < 200:
        raise RuntimeError(f"invalid FIFA ranking response for {schedule_id}")
    return rankings, url


def build_official_snapshots(cutoff: date) -> None:
    output: list[dict] = []
    for year, policy, selected in select_official_snapshots(available_fifa_dates(), cutoff):
        rankings, source_url = fetch_official_ranking(selected["id"])
        for row in rankings:
            output.append(
                {
                    "year": year,
                    "snapshot_policy": policy,
                    "snapshot_date": selected["date"],
                    "date_id": selected["id"],
                    "rank": "" if row.get("rank") is None else row["rank"],
                    "team": build_fifa_annual_rankings.normalized_team(row["teamName"]),
                    "team_raw": row["teamName"],
                    "team_short": row["countryCode"],
                    "total_points": f"{float(row['totalPoints']):.6f}",
                    "confederation": row.get("confederationName", ""),
                    "source": "FIFA official API",
                    "source_url": source_url,
                }
            )
    fields = [
        "year",
        "snapshot_policy",
        "snapshot_date",
        "date_id",
        "rank",
        "team",
        "team_raw",
        "team_short",
        "total_points",
        "confederation",
        "source",
        "source_url",
    ]
    write_csv(OFFICIAL_SNAPSHOTS_CSV, fields, output)
    dates = sorted({row["snapshot_date"] for row in output})
    print(f"Wrote {OFFICIAL_SNAPSHOTS_CSV} ({', '.join(dates)})")


def world_cup_teams() -> list[str]:
    return sorted({canonical_team(team) for match in schedule() for team in (match.team_a, match.team_b)})


def build_squad_value_proxy(min_value: float, max_value: float) -> None:
    rankings: dict[str, float] = {}
    with ANNUAL_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["year"] == "2026":
                rankings[canonical_team(row["team"])] = float(row["total_points"])

    teams = world_cup_teams()
    missing = [team for team in teams if team not in rankings]
    if missing:
        raise RuntimeError(f"missing 2026 FIFA points for proxy teams: {', '.join(missing)}")
    points = [rankings[team] for team in teams]
    low, high = min(points), max(points)
    if low == high:
        raise RuntimeError("cannot build squad-value proxy from identical FIFA points")

    rows: list[dict] = []
    for team in teams:
        ratio = (rankings[team] - low) / (high - low)
        total = math.exp(math.log(min_value) + ratio * (math.log(max_value) - math.log(min_value)))
        rows.append(
            {
                "team": team,
                "squad_size": 26,
                "average_age": "",
                "world_cup_participations": "",
                "foreigners_percent": "",
                "market_value_eur_m": f"{total:.3f}",
                "average_market_value_eur_m": f"{total / 26:.3f}",
                "source": PUBLIC_PROXY_MARKER,
            }
        )
    fields = [
        "team",
        "squad_size",
        "average_age",
        "world_cup_participations",
        "foreigners_percent",
        "market_value_eur_m",
        "average_market_value_eur_m",
        "source",
    ]
    write_csv(SQUAD_PROXY_CSV, fields, rows)
    print(f"Wrote {SQUAD_PROXY_CSV} (public proxy, not Transfermarkt data)")


def initialize_optional_key_player_signals() -> None:
    if KEY_PLAYER_SIGNALS_CSV.exists():
        return
    schema = SCHEMA_DIR / KEY_PLAYER_SIGNALS_CSV.name
    KEY_PLAYER_SIGNALS_CSV.write_bytes(schema.read_bytes())
    print(f"Wrote {KEY_PLAYER_SIGNALS_CSV} (optional signal disabled)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reproducible public model inputs.")
    parser.add_argument("--cutoff", type=date.fromisoformat, default=PRE_TOURNAMENT_CUTOFF)
    parser.add_argument("--min-value", type=float, default=30.0)
    parser.add_argument("--max-value", type=float, default=1300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cutoff > PRE_TOURNAMENT_CUTOFF:
        raise RuntimeError(
            f"cutoff {args.cutoff} is after the fixed pre-tournament date {PRE_TOURNAMENT_CUTOFF}"
        )
    if not 0 < args.min_value < args.max_value:
        raise RuntimeError("proxy values require 0 < min-value < max-value")
    refuse_to_overwrite_user_inputs()
    build_open_history()
    build_official_snapshots(args.cutoff)
    build_fifa_annual_rankings.main()
    build_squad_value_proxy(args.min_value, args.max_value)
    initialize_optional_key_player_signals()


if __name__ == "__main__":
    main()
