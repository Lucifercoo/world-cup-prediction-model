from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "output" / "realtime_context_adjusted_plan.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect generated match predictions.")
    parser.add_argument("--date", help="Filter by Beijing date (YYYY-MM-DD).")
    parser.add_argument("--team", help="Filter by English team name.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def outcome(row: dict[str, str]) -> str:
    return {"A": f"{row['team_a']} win", "D": "Draw", "B": f"{row['team_b']} win"}[
        row["predicted_outcome"]
    ]


def select_rows(path: Path, match_date: str | None, team: str | None) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"{path} does not exist; run `uv run python -m wc_model setup` first")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if match_date:
        rows = [row for row in rows if row["date_bjt"] == match_date]
    if team:
        normalized = team.casefold()
        rows = [
            row
            for row in rows
            if normalized in {row["team_a"].casefold(), row["team_b"].casefold()}
        ]
    return sorted(rows, key=lambda row: (row["date_bjt"], row["time_bjt"], row["team_a"]))


def render(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No predictions matched the filters."
    lines = [
        "date/time (BJT) | match | outcome | goals Top1/Top2 | model | backup | value | upset",
        "-" * 122,
    ]
    for row in rows:
        lines.append(
            f"{row['date_bjt']} {row['time_bjt']} | {row['team_a']} vs {row['team_b']} | "
            f"{outcome(row)} | {row['adjusted_total_goal_bucket']}/{row['backup_total_goal_bucket']} | "
            f"{row['adjusted_score_1_model']} | {row['adjusted_score_2_aggressive_prediction']} | "
            f"{row['adjusted_score_3_market_value']} | {row['adjusted_score_4_upset']}"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    print(render(select_rows(args.input, args.date, args.team)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
