from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

from predict import DATA_DIR, OUTPUT_DIR, RESULTS_CSV, canonical_team, download_results, parse_result_date
from profiles import (
    HALF_LIFE_DAYS,
    MIN_COMPARISON_SAMPLE_SIZE,
    MIN_COMPARISON_WEIGHTED_SAMPLE_SIZE,
    ProfileConfig,
    build_profiles,
    load_results_window,
)
from style_matchups import STYLE_MATCHUP_EDGES_CSV, profile_row_style_features, write_feature_snapshot


START_YEAR = 2016
END_DATE = date(2026, 6, 11)
ROLLING_YEARS = 10
SNAPSHOT_CUTOFF_MONTH = 1
SNAPSHOT_CUTOFF_DAY = 1
SHRINK_WEIGHT = 70.0
COMPETITIVE_ONLY = True
OUTPUT_SUMMARY = OUTPUT_DIR / "style_matchup_edges_summary.md"


EXCLUDED_TOURNAMENTS = {
    "Friendly",
    "Unofficial",
}


@dataclass
class EdgeAccumulator:
    weighted_matches: float = 0.0
    actual_share: float = 0.0
    expected_share: float = 0.0
    goal_diff: float = 0.0
    expected_goal_diff: float = 0.0
    total_goals: float = 0.0


def load_all_results() -> list[dict]:
    download_results()
    rows: list[dict] = []
    with RESULTS_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            match_date = parse_result_date(row["date"])
            if match_date < date(START_YEAR, 1, 1) or match_date > END_DATE:
                continue
            if not row.get("home_score", "").isdigit() or not row.get("away_score", "").isdigit():
                continue
            tournament = row.get("tournament", "")
            if COMPETITIVE_ONLY and tournament in EXCLUDED_TOURNAMENTS:
                continue
            rows.append(
                {
                    "date": match_date,
                    "home_team": canonical_team(row["home_team"]),
                    "away_team": canonical_team(row["away_team"]),
                    "home_score": int(row["home_score"]),
                    "away_score": int(row["away_score"]),
                    "neutral": row.get("neutral", "FALSE").upper() == "TRUE",
                    "tournament": tournament,
                }
            )
    rows.sort(key=lambda item: item["date"])
    return rows


def teams_in_results(results: list[dict]) -> frozenset[str]:
    teams = set()
    for row in results:
        teams.add(row["home_team"])
        teams.add(row["away_team"])
    return frozenset(teams)


def build_snapshot(year: int) -> dict[str, dict]:
    cutoff = date(year, SNAPSHOT_CUTOFF_MONTH, SNAPSHOT_CUTOFF_DAY)
    start = date(cutoff.year - ROLLING_YEARS, cutoff.month, cutoff.day)
    window = load_results_window(start, cutoff)
    config = ProfileConfig(
        year=year,
        start_date=start,
        cutoff_date=cutoff,
        target_teams=teams_in_results(window),
        output_stem=f"_style_matchup_profiles_{year}",
        target_label="历史风格克制样本",
    )
    profiles = build_profiles(window, config)
    return {
        row["team"]: row
        for row in profiles
        if row["sample_size"] >= MIN_COMPARISON_SAMPLE_SIZE
        and row["weighted_sample_size"] >= MIN_COMPARISON_WEIGHTED_SAMPLE_SIZE
    }


def actual_share(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def expected_share(profile_a: dict, profile_b: dict, *, home_for_a: bool, neutral: bool) -> float:
    strength_edge = float(profile_a["strength_score"]) - float(profile_b["strength_score"])
    home_edge = 0.0 if neutral else (0.10 if home_for_a else -0.10)
    return 1.0 / (1.0 + math.exp(-(strength_edge + home_edge) / 0.52))


def expected_goal_diff(profile_a: dict, profile_b: dict, *, home_for_a: bool, neutral: bool) -> float:
    strength_edge = float(profile_a["strength_score"]) - float(profile_b["strength_score"])
    home_edge = 0.0 if neutral else (0.10 if home_for_a else -0.10)
    return (strength_edge + home_edge) * 0.95


def match_weight(match_date: date) -> float:
    age_days = max(0, (END_DATE - match_date).days)
    return 0.5 ** (age_days / (HALF_LIFE_DAYS * 1.35))


def add_edge(
    edges: dict[tuple[str, str], EdgeAccumulator],
    feature_a: str,
    feature_b: str,
    *,
    weight: float,
    actual: float,
    expected: float,
    goal_diff: float,
    expected_gd: float,
    total_goals: int,
) -> None:
    edge = edges[(feature_a, feature_b)]
    edge.weighted_matches += weight
    edge.actual_share += actual * weight
    edge.expected_share += expected * weight
    edge.goal_diff += goal_diff * weight
    edge.expected_goal_diff += expected_gd * weight
    edge.total_goals += total_goals * weight


def build_edges() -> tuple[dict[tuple[str, str], EdgeAccumulator], dict[int, dict[str, dict]], int]:
    results = load_all_results()
    snapshots = {year: build_snapshot(year) for year in range(START_YEAR, END_DATE.year + 1)}
    edges: dict[tuple[str, str], EdgeAccumulator] = defaultdict(EdgeAccumulator)
    used_matches = 0
    for row in results:
        snapshot = snapshots[row["date"].year]
        home = row["home_team"]
        away = row["away_team"]
        if home not in snapshot or away not in snapshot:
            continue
        profile_home = snapshot[home]
        profile_away = snapshot[away]
        features_home = profile_row_style_features(profile_home)
        features_away = profile_row_style_features(profile_away)
        if not features_home or not features_away:
            continue
        weight = match_weight(row["date"])
        home_actual = actual_share(row["home_score"], row["away_score"])
        away_actual = 1.0 - home_actual if home_actual != 0.5 else 0.5
        home_expected = expected_share(profile_home, profile_away, home_for_a=True, neutral=row["neutral"])
        away_expected = expected_share(profile_away, profile_home, home_for_a=False, neutral=row["neutral"])
        home_expected_gd = expected_goal_diff(profile_home, profile_away, home_for_a=True, neutral=row["neutral"])
        away_expected_gd = -home_expected_gd
        home_gd = row["home_score"] - row["away_score"]
        away_gd = -home_gd
        total_goals = row["home_score"] + row["away_score"]
        for feature_home in features_home:
            for feature_away in features_away:
                add_edge(
                    edges,
                    feature_home,
                    feature_away,
                    weight=weight,
                    actual=home_actual,
                    expected=home_expected,
                    goal_diff=home_gd,
                    expected_gd=home_expected_gd,
                    total_goals=total_goals,
                )
                add_edge(
                    edges,
                    feature_away,
                    feature_home,
                    weight=weight,
                    actual=away_actual,
                    expected=away_expected,
                    goal_diff=away_gd,
                    expected_gd=away_expected_gd,
                    total_goals=total_goals,
                )
        used_matches += 1
    return edges, snapshots, used_matches


def write_edges(edges: dict[tuple[str, str], EdgeAccumulator], used_matches: int) -> None:
    rows: list[dict] = []
    for (feature_a, feature_b), edge in edges.items():
        if edge.weighted_matches <= 0:
            continue
        actual = edge.actual_share / edge.weighted_matches
        expected = edge.expected_share / edge.weighted_matches
        residual = actual - expected
        goal_diff_residual = edge.goal_diff / edge.weighted_matches - edge.expected_goal_diff / edge.weighted_matches
        average_total_goals = edge.total_goals / edge.weighted_matches
        shrunk = residual * edge.weighted_matches / (edge.weighted_matches + SHRINK_WEIGHT)
        total_multiplier = max(0.90, min(1.14, average_total_goals / 2.55))
        rows.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "weighted_matches": edge.weighted_matches,
                "actual_points_share_a": actual,
                "expected_points_share_a": expected,
                "residual_points_share_a": residual,
                "shrunk_residual": shrunk,
                "goal_diff_residual_a": goal_diff_residual,
                "average_total_goals": average_total_goals,
                "total_goal_multiplier": total_multiplier,
            }
        )
    rows.sort(
        key=lambda row: (
            -round(abs(row["shrunk_residual"]), 6),
            row["feature_a"],
            row["feature_b"],
        )
    )
    fieldnames = [
        "feature_a",
        "feature_b",
        "weighted_matches",
        "actual_points_share_a",
        "expected_points_share_a",
        "residual_points_share_a",
        "shrunk_residual",
        "goal_diff_residual_a",
        "average_total_goals",
        "total_goal_multiplier",
    ]
    with STYLE_MATCHUP_EDGES_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )

    summary_rows = [
        row for row in rows if row["weighted_matches"] >= 35.0 and abs(row["shrunk_residual"]) >= 0.010
    ][:30]
    lines = [
        "# Style Matchup Edges",
        "",
        f"- Historical matches used: {used_matches}",
        f"- Output: `{STYLE_MATCHUP_EDGES_CSV}`",
        "",
        "| Feature A | Feature B | Weighted matches | Residual | Total mult |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {a} | {b} | {n:.1f} | {r:+.3f} | {t:.3f} |".format(
                a=row["feature_a"],
                b=row["feature_b"],
                n=row["weighted_matches"],
                r=row["shrunk_residual"],
                t=row["total_goal_multiplier"],
            )
        )
    OUTPUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    edges, snapshots, used_matches = build_edges()
    write_edges(edges, used_matches)
    current_profiles = snapshots[END_DATE.year]
    write_feature_snapshot(list(current_profiles.values()))
    print(f"Style matchup edges: {STYLE_MATCHUP_EDGES_CSV}")
    print(f"Summary: {OUTPUT_SUMMARY}")
    print(f"Historical matches used: {used_matches}")


if __name__ == "__main__":
    main()
