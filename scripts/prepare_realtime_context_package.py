from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

TEAM_FIELDS = [
    "date_bjt", "time_bjt", "match", "team", "home_adaptation_multiplier",
    "travel_multiplier", "weather_multiplier", "cohesion_multiplier",
    "injury_multiplier", "opponent_attack_multiplier", "tempo_multiplier",
    "weather_high_c", "travel_km", "analysis_notes", "source_urls",
]
SHAPE_FIELDS = [
    "match", "pre_match_shapes", "observed_shapes", "draw_multiplier",
    "tempo_multiplier", "favorite_attack_multiplier", "underdog_attack_multiplier", "notes",
]
KEY_PLAYER_FIELDS = [
    "date_bjt", "time_bjt", "match", "team", "key_player", "status", "impact",
    "source_urls", "notes",
]
MULTIPLIER_RANGES = {
    "home_adaptation_multiplier": (1.00, 1.08),
    "travel_multiplier": (0.94, 1.02),
    "weather_multiplier": (0.94, 1.03),
    "cohesion_multiplier": (0.96, 1.03),
    "injury_multiplier": (0.90, 1.03),
    "opponent_attack_multiplier": (0.94, 1.08),
    "tempo_multiplier": (0.92, 1.08),
}
SHAPE_RANGES = {
    "draw_multiplier": (0.80, 1.25),
    "tempo_multiplier": (0.85, 1.15),
    "favorite_attack_multiplier": (0.80, 1.15),
    "underdog_attack_multiplier": (0.85, 1.20),
}
ALLOWED_SHAPES = {
    "collapse_risk", "controlled_favorite", "credible_opponent", "final",
    "heat_slowdown", "home_pressure", "low_block", "low_conversion", "low_event",
    "low_event_favorite", "low_stakes", "must_win", "open_favorite", "open_game",
    "open_mismatch", "rotation", "set_piece_risk", "star_finishing",
    "star_playmaking", "transition_dog",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and convert LLM-collected realtime context.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def require_keys(value: dict, expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"{label} fields differ: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")


def validate_range(name: str, value: float | None, ranges: dict[str, tuple[float, float]]) -> None:
    if value is None:
        return
    low, high = ranges[name]
    if not low <= float(value) <= high:
        raise ValueError(f"{name}={value} is outside [{low}, {high}]")


def source_text(sources: list[dict]) -> str:
    if not sources:
        raise ValueError("each evidence-bearing record needs at least one source")
    parts = []
    for source in sources:
        require_keys(source, {"label", "url", "published_at"}, "source")
        parsed = urlparse(source["url"])
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"invalid source URL: {source['url']}")
        parts.append(f"{source['label']}:{source['url']}")
    return "; ".join(parts)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_and_convert(payload: dict, output_dir: Path) -> None:
    require_keys(payload, {"collector", "match", "teams", "shape", "key_players"}, "root")
    collector = payload["collector"]
    require_keys(collector, {"model", "reasoning_effort", "collected_at_utc"}, "collector")
    match = payload["match"]
    require_keys(match, {"date_bjt", "time_bjt", "team_a", "team_b", "venue"}, "match")
    collected_at = datetime.fromisoformat(collector["collected_at_utc"].replace("Z", "+00:00"))
    if collected_at.tzinfo is None:
        raise ValueError("collected_at_utc must include a timezone")
    kickoff = datetime.strptime(
        f"{match['date_bjt']} {match['time_bjt']} +0800", "%Y-%m-%d %H:%M %z"
    ).astimezone(timezone.utc)
    if collected_at.astimezone(timezone.utc) >= kickoff:
        raise ValueError("realtime context must be collected before kickoff")

    match_name = f"{match['team_a']} vs {match['team_b']}"
    if len(payload["teams"]) != 2 or {row["team"] for row in payload["teams"]} != {
        match["team_a"], match["team_b"]
    }:
        raise ValueError("teams must contain exactly team_a and team_b")

    team_rows = []
    for team in payload["teams"]:
        require_keys(
            team,
            {"team", "multipliers", "weather_high_c", "travel_km", "analysis_notes", "sources"},
            "team",
        )
        require_keys(team["multipliers"], set(MULTIPLIER_RANGES), "multipliers")
        for name, value in team["multipliers"].items():
            validate_range(name, value, MULTIPLIER_RANGES)
        team_rows.append({
            "date_bjt": match["date_bjt"], "time_bjt": match["time_bjt"],
            "match": match_name, "team": team["team"],
            **{name: "" if value is None else value for name, value in team["multipliers"].items()},
            "weather_high_c": "" if team["weather_high_c"] is None else team["weather_high_c"],
            "travel_km": "" if team["travel_km"] is None else team["travel_km"],
            "analysis_notes": team["analysis_notes"], "source_urls": source_text(team["sources"]),
        })

    shape = payload["shape"]
    require_keys(
        shape,
        {"pre_match_shapes", "draw_multiplier", "tempo_multiplier", "favorite_attack_multiplier", "underdog_attack_multiplier", "notes", "sources"},
        "shape",
    )
    for name in SHAPE_RANGES:
        validate_range(name, shape[name], SHAPE_RANGES)
    unknown_shapes = set(shape["pre_match_shapes"]) - ALLOWED_SHAPES
    if unknown_shapes:
        raise ValueError(f"unknown pre-match shapes: {sorted(unknown_shapes)}")
    shape_sources = source_text(shape["sources"]) if shape["sources"] else ""
    shape_row = {
        "match": match_name,
        "pre_match_shapes": ";".join(shape["pre_match_shapes"]),
        "observed_shapes": "",
        "draw_multiplier": shape["draw_multiplier"],
        "tempo_multiplier": shape["tempo_multiplier"],
        "favorite_attack_multiplier": shape["favorite_attack_multiplier"],
        "underdog_attack_multiplier": shape["underdog_attack_multiplier"],
        "notes": f"{shape['notes']} Sources: {shape_sources}" if shape_sources else shape["notes"],
    }

    key_rows = []
    for player in payload["key_players"]:
        require_keys(player, {"team", "key_player", "status", "impact", "notes", "sources"}, "key_player")
        if player["team"] not in {match["team_a"], match["team_b"]}:
            raise ValueError(f"key player has unknown team: {player['team']}")
        if not -1.0 <= float(player["impact"]) <= 1.0:
            raise ValueError("key-player impact must be within [-1, 1]")
        key_rows.append({
            "date_bjt": match["date_bjt"], "time_bjt": match["time_bjt"],
            "match": match_name, "team": player["team"], "key_player": player["key_player"],
            "status": player["status"], "impact": player["impact"],
            "source_urls": source_text(player["sources"]), "notes": player["notes"],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "realtime_team_context.csv", TEAM_FIELDS, team_rows)
    write_csv(output_dir / "match_shape_context.csv", SHAPE_FIELDS, [shape_row])
    write_csv(output_dir / "world_cup_2026_key_player_match_status.csv", KEY_PLAYER_FIELDS, key_rows)
    (output_dir / "context_package.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    validate_and_convert(payload, args.output_dir)
    print(f"Validated context package: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
