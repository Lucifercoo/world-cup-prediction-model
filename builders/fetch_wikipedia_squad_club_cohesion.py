from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

from predict import DATA_DIR, canonical_team


SQUADS_REVISION = 1359407494
SQUADS_URL = f"https://en.wikipedia.org/w/index.php?title=2026_FIFA_World_Cup_squads&oldid={SQUADS_REVISION}"

TEAM_ALIASES = {
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cape Verde",
    "Czech Republic": "Czechia",
    "DR Congo": "DR Congo",
    "Ivory Coast": "Ivory Coast",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "Türkiye": "Turkey",
    "Turkey": "Turkey",
    "United States": "United States",
}


def fetch_html() -> str:
    req = urllib.request.Request(
        SQUADS_URL,
        headers={"User-Agent": "codex-wc2026-squad-cohesion/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", "replace")


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def canonical_squad_team(name: str) -> str:
    return canonical_team(TEAM_ALIASES.get(clean_text(name), clean_text(name)))


def table_team_name(table) -> str:
    heading = table.find_previous(["h3", "h2"])
    if heading is None:
        raise RuntimeError("squad table has no previous heading")
    headline = clean_text(heading.get_text(" ", strip=True))
    headline = re.sub(r"\s*\[ edit \]\s*$", "", headline)
    return canonical_squad_team(headline)


def club_from_cell(cell) -> str:
    links = cell.find_all("a")
    if links:
        text = clean_text(links[-1].get_text(" ", strip=True))
        if text:
            return text
    return clean_text(cell.get_text(" ", strip=True))


def parse_players(source_html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(source_html, "html.parser")
    players: list[dict[str, str]] = []
    for table in soup.find_all("table", class_="wikitable"):
        header_cells = [clean_text(th.get_text(" ", strip=True)) for th in table.find_all("th", recursive=True)[:7]]
        if header_cells[:7] != ["No.", "Pos.", "Player", "Date of birth (age)", "Caps", "Goals", "Club"]:
            continue
        team = table_team_name(table)
        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"], recursive=False)
            if len(cells) < 7:
                continue
            player_cell = cells[2]
            club_cell = cells[6]
            players.append(
                {
                    "team": team,
                    "squad_no": clean_text(cells[0].get_text(" ", strip=True)),
                    "position": clean_text(cells[1].get_text(" ", strip=True)).replace("1 ", "").replace("2 ", "").replace("3 ", "").replace("4 ", ""),
                    "player": clean_text(player_cell.get_text(" ", strip=True)),
                    "club": club_from_cell(club_cell),
                    "source": SQUADS_URL,
                }
            )
    if len(players) != 48 * 26:
        raise RuntimeError(f"expected 1248 squad players, got {len(players)}")
    return players


def cohesion_multiplier(max_club_share: float, top3_club_share: float) -> float:
    multiplier = 1.0
    if max_club_share >= 0.30:
        multiplier += 0.03
    elif max_club_share >= 0.23:
        multiplier += 0.02
    if top3_club_share >= 0.50:
        multiplier += 0.02
    return round(multiplier, 4)


def build_cohesion_rows(players: list[dict[str, str]]) -> list[dict[str, str]]:
    by_team: dict[str, list[dict[str, str]]] = {}
    for player in players:
        by_team.setdefault(player["team"], []).append(player)
    if len(by_team) != 48:
        raise RuntimeError(f"expected 48 teams, got {len(by_team)}")

    rows: list[dict[str, str]] = []
    for team in sorted(by_team):
        team_players = by_team[team]
        if len(team_players) != 26:
            raise RuntimeError(f"expected 26 players for {team}, got {len(team_players)}")
        counts = Counter(player["club"] for player in team_players)
        ordered = counts.most_common()
        max_count = ordered[0][1]
        top3_count = sum(count for _, count in ordered[:3])
        max_share = max_count / len(team_players)
        top3_share = top3_count / len(team_players)
        rows.append(
            {
                "team": team,
                "squad_size": str(len(team_players)),
                "top_club": ordered[0][0],
                "top_club_players": str(max_count),
                "top3_club_players": str(top3_count),
                "max_club_share": f"{max_share:.4f}",
                "top3_club_share": f"{top3_share:.4f}",
                "club_cohesion_multiplier": f"{cohesion_multiplier(max_share, top3_share):.4f}",
                "club_counts": "; ".join(f"{club}:{count}" for club, count in ordered),
                "source": SQUADS_URL,
            }
        )
    return rows


def write_csv(path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build squad-club and cohesion datasets from a fixed Wikipedia revision.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = parse_args()
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    players_csv = data_dir / "world_cup_2026_squad_clubs.csv"
    cohesion_csv = data_dir / "world_cup_2026_team_club_cohesion.csv"
    players = parse_players(fetch_html())
    cohesion_rows = build_cohesion_rows(players)
    write_csv(
        players_csv,
        players,
        ["team", "squad_no", "position", "player", "club", "source"],
    )
    write_csv(
        cohesion_csv,
        cohesion_rows,
        [
            "team",
            "squad_size",
            "top_club",
            "top_club_players",
            "top3_club_players",
            "max_club_share",
            "top3_club_share",
            "club_cohesion_multiplier",
            "club_counts",
            "source",
        ],
    )
    print(f"Wrote {players_csv}")
    print(f"Wrote {cohesion_csv}")
    print(f"Teams: {len(cohesion_rows)}")


if __name__ == "__main__":
    main()
