from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from build_fifa_annual_rankings import normalized_team
from predict import DATA_DIR, OUTPUT_DIR, Match, canonical_team, host_multiplier, schedule
from style_matchups import (
    apply_style_influence_gate,
    profile_style_features,
    style_influence_factor,
    style_matchup_effect,
)


FIFA_RANKING_CSV = DATA_DIR / "fifa_rankings_annual_start.csv"
LIVE_RANKING_CSV = OUTPUT_DIR / "world_cup_2026_live_rankings.csv"
MARKET_VALUE_CSV = DATA_DIR / "transfermarkt_world_cup_2026_values.csv"
CLUB_COHESION_CSV = DATA_DIR / "world_cup_2026_team_club_cohesion.csv"
IN_TOURNAMENT_ADJUSTMENTS_CSV = DATA_DIR / "in_tournament_team_adjustments.csv"
PROFILE_CSV = OUTPUT_DIR / "team_profiles_2026.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "group_score_predictions_fifa_profile.csv"
PREDICTIONS_MD = OUTPUT_DIR / "group_score_predictions_fifa_profile.md"
RANKING_YEAR = 2026
WORLD_CUP_GROUP_GOALS_PER_MATCH = 2.4947916666666665
MAX_GOALS = 8
DRAW_RANK_GAP = 10
POINT_EDGE_SCALE = 180.0
MARKET_VALUE_OUTCOME_EDGE_WEIGHT = 42.0
MARKET_VALUE_XG_SPLIT_WEIGHT = 0.12
MARKET_VALUE_TEMPO_WEIGHT = 0.08
MARKET_VALUE_ATTACK_WEIGHT = 0.10
MARKET_VALUE_SIGNAL_SCALE = 2.4
DRAW_BASE = 0.24
DRAW_CLOSE_BONUS = 0.08
DRAW_STYLE_BONUS = 0.05
CLOSE_OUTCOME_MARGIN = 0.08
TOTAL_GOAL_BUCKET_LABELS = ("0-1球", "2-3球", "4-5球", "6-8球")
BASE_TOTAL_GOAL_BUCKET_PROBABILITIES = {
    "0-1球": 0.25,
    "2-3球": 0.34,
    "4-5球": 0.28,
    "6-8球": 0.13,
}
BASE_TOTAL_GOAL_MID_BUCKET_FACTOR = 0.90
TOTAL_GOAL_BUCKET_ADJUSTMENT_STRENGTH = 0.50
TOP20_ATTACK_MULTIPLIER = 1.08
TOP20_VS_NON_TOP20_TEMPO_MULTIPLIER = 1.05
MISMATCH_TEMPO_POINT_SCALE = 260.0
MISMATCH_ATTACK_POINT_SCALE = 320.0
MISMATCH_MAX_TEMPO_MULTIPLIER = 1.32
MISMATCH_MAX_ATTACK_MULTIPLIER = 1.28
AGGRESSIVE_TOP20_ATTACK_MULTIPLIER = 1.55
AGGRESSIVE_TOP20_VS_NON_TOP20_TEMPO_MULTIPLIER = 1.20
MARKET_VALUE_SHARE_WEIGHT = 0.80
FIFA_RANK_SHARE_WEIGHT = 1.20
FIFA_RANK_SHARE_SCALE = 45.0
MARKET_VALUE_SCORE_MAX_TOTAL_GOALS = 6
MARKET_VALUE_EXTREME_MISMATCH_MAX_TOTAL_GOALS = 8
MARKET_VALUE_HIGH_BUCKET_PROMOTION_RATIO = 0.70
MARKET_VALUE_HIGH_BUCKET_MAX_TOP_PROBABILITY = 0.50
HIGH_BUCKET_COVERAGE_RATIO = 0.55
HIGH_BUCKET_COVERAGE_MIN_PROBABILITY = 0.20
COLLAPSE_SIX_PLUS_MIN_PROBABILITY = 0.025
HIGH_TOTAL_GOAL_BUCKETS = {"4-5球", "6-8球"}
OPEN_TOTAL_GOAL_SHAPES = {"open_game", "open_mismatch", "transition_dog", "set_piece_risk"}
LOW_TOTAL_GOAL_SHAPES = {"low_block", "low_event_favorite"}
EXTREME_TOTAL_GOAL_SHAPES = {"collapse_risk"}
UNDERDOG_GOAL_TOTAL_THRESHOLD = 4
UNDERDOG_GOAL_MIN_XG = 0.45
UNDERDOG_GOAL_MIN_OUTCOME_PROBABILITY = 0.10
UNDERDOG_GOAL_MAX_WIN_PROBABILITY = 0.72
UNDERDOG_GOAL_MAX_XG_EDGE = 0.78
OPEN_FAVORITE_UNDERDOG_XG_MAX_MULTIPLIER = 1.16
OPEN_FAVORITE_UNDERDOG_XG_WEIGHT = 0.28
IN_TOURNAMENT_EDGE_COMPRESSION_WEIGHT = 0.08


def require_input(path: Path, description: str) -> None:
    if not path.exists():
        raise RuntimeError(
            f"missing required {description}: {path}. "
            "See docs/DATA_FETCH.csv and docs/DATA_SOURCES.md for acquisition requirements."
        )
IN_TOURNAMENT_EDGE_COMPRESSION_LIMIT = 0.22
IN_TOURNAMENT_EDGE_SHIFT_POINTS = 10.0
IN_TOURNAMENT_DRAW_BOTH_UNDER_WEIGHT = 0.035
IN_TOURNAMENT_DRAW_DIFF_WEIGHT = 0.035
IN_TOURNAMENT_TEMPO_STATE_WEIGHT = 0.55
IN_TOURNAMENT_TEMPO_UNDER_WEIGHT = 0.09
IN_TOURNAMENT_SPLIT_SIGNAL_WEIGHT = 0.03
IN_TOURNAMENT_SPLIT_ATTACK_WEIGHT = 0.02
IN_TOURNAMENT_GOAL_SUPPRESSION_STRONG = 0.48
IN_TOURNAMENT_GOAL_SUPPRESSION_MEDIUM = 0.28
IN_TOURNAMENT_LOW_BUCKET_SUPPRESSION = 0.92
IN_TOURNAMENT_LOW_BUCKET_DRAW = 0.37
UPSET_SCORE_STRONG_FAVORITE_PROBABILITY = 0.70
UPSET_SCORE_MODERATE_FAVORITE_PROBABILITY = 0.55


LOW_EVENT_STYLES = {"防守型", "低效型"}
HIGH_EVENT_STYLES = {"开放型", "进攻型"}
STRONG_CONTROL_STYLES = {"攻守兼备型"}


@dataclass(frozen=True)
class FifaRanking:
    rank: int
    points: float
    snapshot_date: str


@dataclass(frozen=True)
class TeamProfile:
    style: str
    goals_for: float
    goals_against: float
    clean_sheet_rate: float
    multi_goal_rate: float
    conceded_multi_rate: float
    high_total_goal_rate: float
    both_score_rate: float


@dataclass(frozen=True)
class MarketValue:
    total_eur_m: float
    average_eur_m: float


def team_profile_features(profile: TeamProfile) -> frozenset[str]:
    return profile_style_features(
        style=profile.style,
        goals_for=profile.goals_for,
        goals_against=profile.goals_against,
        clean_sheet_rate=profile.clean_sheet_rate,
        multi_goal_rate=profile.multi_goal_rate,
        conceded_multi_rate=profile.conceded_multi_rate,
        high_total_goal_rate=profile.high_total_goal_rate,
        both_score_rate=profile.both_score_rate,
    )


@dataclass(frozen=True)
class ClubCohesion:
    top_club: str
    top_club_players: int
    max_club_share: float
    top3_club_share: float
    multiplier: float


@dataclass(frozen=True)
class ProfileBaselines:
    high_total_goal_rate: float
    both_score_rate: float
    multi_goal_rate: float
    conceded_multi_rate: float


@dataclass(frozen=True)
class InTournamentAdjustment:
    effective_after_bjt: datetime
    points_adjustment: float
    attack_multiplier: float
    tempo_multiplier: float
    reason: str


def load_in_tournament_adjustments() -> dict[str, InTournamentAdjustment]:
    adjustments: dict[str, InTournamentAdjustment] = {}
    if not IN_TOURNAMENT_ADJUSTMENTS_CSV.exists():
        return adjustments
    with IN_TOURNAMENT_ADJUSTMENTS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            team = canonical_team(row["team"])
            adjustments[team] = InTournamentAdjustment(
                effective_after_bjt=datetime.strptime(row["effective_after_bjt"], "%Y-%m-%d %H:%M"),
                points_adjustment=float(row["points_adjustment"]),
                attack_multiplier=float(row["attack_multiplier"]),
                tempo_multiplier=float(row["tempo_multiplier"]),
                reason=row["reason"],
            )
    return adjustments


IN_TOURNAMENT_ADJUSTMENTS = load_in_tournament_adjustments()


def load_fifa_rankings() -> dict[str, FifaRanking]:
    rankings: dict[str, FifaRanking] = {}
    if LIVE_RANKING_CSV.exists():
        with LIVE_RANKING_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                team = canonical_team(row["team"])
                rankings[team] = FifaRanking(
                    rank=int(row["live_rank"]),
                    points=float(row["live_points"]),
                    snapshot_date=f"live:{row['snapshot_date']}",
                )
        if len(rankings) != 48:
            raise RuntimeError(f"expected 48 live World Cup rankings, got {len(rankings)}")
        return rankings

    require_input(FIFA_RANKING_CSV, "FIFA ranking input")
    with FIFA_RANKING_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["year"]) != RANKING_YEAR:
                continue
            team = normalized_team(row["team"])
            rankings[team] = FifaRanking(
                rank=int(row["rank"]),
                points=float(row["total_points"]),
                snapshot_date=row["snapshot_date"],
            )
    if len(rankings) < 200:
        raise RuntimeError(f"expected at least 200 FIFA rankings for {RANKING_YEAR}, got {len(rankings)}")
    return rankings


def load_profiles() -> dict[str, TeamProfile]:
    profiles: dict[str, TeamProfile] = {}
    require_input(PROFILE_CSV, "generated team profile")
    with PROFILE_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            team = canonical_team(row["team"])
            profiles[team] = TeamProfile(
                style=row["style"],
                goals_for=float(row["weighted_goals_for"]),
                goals_against=float(row["weighted_goals_against"]),
                clean_sheet_rate=float(row["clean_sheet_rate"]),
                multi_goal_rate=float(row["multi_goal_rate"]),
                conceded_multi_rate=float(row["conceded_multi_rate"]),
                high_total_goal_rate=float(row["high_total_goal_rate"]),
                both_score_rate=float(row["both_score_rate"]),
            )
    return profiles


def load_market_values() -> dict[str, MarketValue]:
    values: dict[str, MarketValue] = {}
    require_input(MARKET_VALUE_CSV, "user-supplied squad market-value input")
    with MARKET_VALUE_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            team = canonical_team(row["team"])
            values[team] = MarketValue(
                total_eur_m=float(row["market_value_eur_m"]),
                average_eur_m=float(row["average_market_value_eur_m"]),
            )
    if len(values) != 48:
        raise RuntimeError(f"expected 48 World Cup market values, got {len(values)}")
    return values


def load_club_cohesion() -> dict[str, ClubCohesion]:
    values: dict[str, ClubCohesion] = {}
    require_input(CLUB_COHESION_CSV, "club-cohesion input")
    with CLUB_COHESION_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            team = canonical_team(row["team"])
            values[team] = ClubCohesion(
                top_club=row["top_club"],
                top_club_players=int(row["top_club_players"]),
                max_club_share=float(row["max_club_share"]),
                top3_club_share=float(row["top3_club_share"]),
                multiplier=float(row["club_cohesion_multiplier"]),
            )
    if len(values) != 48:
        raise RuntimeError(f"expected 48 World Cup club cohesion rows, got {len(values)}")
    return values


def assert_inputs_cover_schedule(
    rankings: dict[str, FifaRanking],
    profiles: dict[str, TeamProfile],
    market_values: dict[str, MarketValue],
    club_cohesion: dict[str, ClubCohesion],
    matches: list[Match],
) -> None:
    teams = sorted({canonical_team(team) for match in matches for team in (match.team_a, match.team_b)})
    missing_rankings = [team for team in teams if team not in rankings]
    missing_profiles = [team for team in teams if team not in profiles]
    missing_market_values = [team for team in teams if team not in market_values]
    missing_club_cohesion = [team for team in teams if team not in club_cohesion]
    if missing_rankings:
        raise RuntimeError(f"missing FIFA ranking for: {', '.join(missing_rankings)}")
    if missing_profiles:
        raise RuntimeError(f"missing profile for: {', '.join(missing_profiles)}")
    if missing_market_values:
        raise RuntimeError(f"missing market value for: {', '.join(missing_market_values)}")
    if missing_club_cohesion:
        raise RuntimeError(f"missing club cohesion for: {', '.join(missing_club_cohesion)}")


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def average(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values) / len(values)


def profile_baselines(profiles: list[TeamProfile]) -> ProfileBaselines:
    return ProfileBaselines(
        high_total_goal_rate=average([profile.high_total_goal_rate for profile in profiles]),
        both_score_rate=average([profile.both_score_rate for profile in profiles]),
        multi_goal_rate=average([profile.multi_goal_rate for profile in profiles]),
        conceded_multi_rate=average([profile.conceded_multi_rate for profile in profiles]),
    )


def style_total_modifier(
    profile_a: TeamProfile,
    profile_b: TeamProfile,
    baselines: ProfileBaselines,
) -> float:
    modifier = 1.0
    styles = {profile_a.style, profile_b.style}
    if profile_a.style in LOW_EVENT_STYLES:
        modifier -= 0.08
    if profile_b.style in LOW_EVENT_STYLES:
        modifier -= 0.08
    if profile_a.style in HIGH_EVENT_STYLES:
        modifier += 0.07
    if profile_b.style in HIGH_EVENT_STYLES:
        modifier += 0.07
    if profile_a.style in STRONG_CONTROL_STYLES or profile_b.style in STRONG_CONTROL_STYLES:
        modifier -= 0.02
    if styles <= LOW_EVENT_STYLES:
        modifier -= 0.05
    if "开放型" in styles:
        modifier += 0.06

    pair_high_total = (profile_a.high_total_goal_rate + profile_b.high_total_goal_rate) / 2
    pair_both_score = (profile_a.both_score_rate + profile_b.both_score_rate) / 2
    pair_multi_goal = (profile_a.multi_goal_rate + profile_b.multi_goal_rate) / 2
    pair_conceded_multi = (profile_a.conceded_multi_rate + profile_b.conceded_multi_rate) / 2

    modifier += (pair_high_total - baselines.high_total_goal_rate) * 0.70
    modifier += (pair_both_score - baselines.both_score_rate) * 0.35
    modifier += (pair_multi_goal - baselines.multi_goal_rate) * 0.20
    modifier += (pair_conceded_multi - baselines.conceded_multi_rate) * 0.20
    return clamp(modifier, 0.68, 1.45)


def draw_probability(rank_gap: int, profile_a: TeamProfile, profile_b: TeamProfile) -> float:
    close_bonus = DRAW_CLOSE_BONUS * max(0.0, 1.0 - rank_gap / DRAW_RANK_GAP) if rank_gap <= DRAW_RANK_GAP else 0.0
    style_bonus = 0.0
    if profile_a.style in LOW_EVENT_STYLES:
        style_bonus += DRAW_STYLE_BONUS / 2
    if profile_b.style in LOW_EVENT_STYLES:
        style_bonus += DRAW_STYLE_BONUS / 2
    if profile_a.style == "开放型" or profile_b.style == "开放型":
        style_bonus -= 0.04
    return clamp(DRAW_BASE + close_bonus + style_bonus, 0.16, 0.38)


def market_value_signal(value: MarketValue) -> float:
    return math.log((value.total_eur_m + 1.0) / 150.0)


def market_value_edge_points(value_a: MarketValue, value_b: MarketValue) -> float:
    return (
        math.log((value_a.total_eur_m + 1.0) / (value_b.total_eur_m + 1.0))
        * MARKET_VALUE_OUTCOME_EDGE_WEIGHT
    )


def market_value_pair_tempo_multiplier(value_a: MarketValue, value_b: MarketValue) -> float:
    average_signal = (market_value_signal(value_a) + market_value_signal(value_b)) / 2
    return clamp(1.0 + average_signal / MARKET_VALUE_SIGNAL_SCALE * MARKET_VALUE_TEMPO_WEIGHT, 0.92, 1.16)


def market_value_attack_multiplier(value: MarketValue) -> float:
    return clamp(1.0 + market_value_signal(value) / MARKET_VALUE_SIGNAL_SCALE * MARKET_VALUE_ATTACK_WEIGHT, 0.88, 1.20)


def open_favorite_underdog_multiplier(
    favorite_profile: TeamProfile,
    underdog_profile: TeamProfile,
    favorite_probability: float,
    underdog_probability: float,
) -> float:
    if favorite_probability < 0.46:
        return 1.0
    if underdog_probability < 0.08:
        return 1.0

    favorite_openness = 0.0
    if favorite_profile.style in HIGH_EVENT_STYLES:
        favorite_openness += 0.40
    favorite_openness += max(0.0, favorite_profile.goals_against - 0.75) * 0.18
    favorite_openness += max(0.0, favorite_profile.conceded_multi_rate - 0.16) * 0.75
    favorite_openness += max(0.0, 0.55 - favorite_profile.clean_sheet_rate) * 0.35

    underdog_counter = 0.0
    if underdog_profile.style in HIGH_EVENT_STYLES:
        underdog_counter += 0.30
    if underdog_profile.style == "攻守兼备型":
        underdog_counter += 0.12
    underdog_counter += max(0.0, underdog_profile.goals_for - 1.10) * 0.16
    underdog_counter += max(0.0, underdog_profile.both_score_rate - 0.34) * 0.55
    underdog_counter += max(0.0, underdog_profile.multi_goal_rate - 0.34) * 0.18

    signal = favorite_openness * 0.62 + underdog_counter * 0.38
    return clamp(
        1.0 + signal * OPEN_FAVORITE_UNDERDOG_XG_WEIGHT,
        1.0,
        OPEN_FAVORITE_UNDERDOG_XG_MAX_MULTIPLIER,
    )


def apply_open_favorite_underdog_xg(
    lambda_a: float,
    lambda_b: float,
    profile_a: TeamProfile,
    profile_b: TeamProfile,
    p_a: float,
    p_b: float,
) -> tuple[float, float]:
    if p_a >= p_b:
        lambda_b *= open_favorite_underdog_multiplier(profile_a, profile_b, p_a, p_b)
    else:
        lambda_a *= open_favorite_underdog_multiplier(profile_b, profile_a, p_b, p_a)
    return lambda_a, lambda_b


def outcome_uncertainty(p_a: float, p_draw: float, p_b: float) -> tuple[float, float, float]:
    ordered = sorted([p_a, p_draw, p_b], reverse=True)
    top_probability = ordered[0]
    outcome_margin = ordered[0] - ordered[1]
    uncertainty = 0.0
    uncertainty += (1.0 - top_probability) * 0.65
    uncertainty += (1.0 - min(outcome_margin, 0.35) / 0.35) * 0.25
    uncertainty += min(p_draw, 0.38) / 0.38 * 0.10
    return clamp(uncertainty, 0.0, 1.0), top_probability, outcome_margin


def predicted_outcome_from_probabilities(
    p_a: float,
    p_draw: float,
    p_b: float,
    *,
    home_label: str = "A",
    draw_label: str = "D",
    away_label: str = "B",
) -> str:
    _, _, outcome_margin = outcome_uncertainty(p_a, p_draw, p_b)
    if outcome_margin < CLOSE_OUTCOME_MARGIN:
        return draw_label
    probabilities = {home_label: p_a, draw_label: p_draw, away_label: p_b}
    return max(probabilities, key=probabilities.get)


def risk_label(uncertainty: float) -> str:
    if uncertainty >= 0.62:
        return "高"
    if uncertainty >= 0.50:
        return "中高"
    if uncertainty >= 0.38:
        return "中"
    return "低"


def risk_reasons(
    p_a: float,
    p_draw: float,
    p_b: float,
    rank_gap: int,
    total_goal_mode: int,
) -> str:
    uncertainty, top_probability, outcome_margin = outcome_uncertainty(p_a, p_draw, p_b)
    reasons: list[str] = []
    if outcome_margin < CLOSE_OUTCOME_MARGIN:
        reasons.append("三项概率接近")
    if top_probability < 0.45:
        reasons.append("最高赛果概率偏低")
    if p_draw >= 0.30:
        reasons.append("平局概率高")
    if rank_gap <= DRAW_RANK_GAP:
        reasons.append("FIFA排名接近")
    if total_goal_mode <= 2:
        reasons.append("低比分一球差风险")
    if not reasons:
        reasons.append(f"不确定性{risk_label(uncertainty)}")
    return "；".join(reasons)


def outcome_probabilities(
    match: Match,
    rankings: dict[str, FifaRanking],
    profiles: dict[str, TeamProfile],
    market_values: dict[str, MarketValue],
) -> tuple[float, float, float]:
    team_a = canonical_team(match.team_a)
    team_b = canonical_team(match.team_b)
    rank_a = rankings[team_a]
    rank_b = rankings[team_b]
    profile_a = profiles[team_a]
    profile_b = profiles[team_b]
    value_a = market_values[team_a]
    value_b = market_values[team_b]

    raw_point_edge = rank_a.points - rank_b.points
    point_edge = adjusted_point_edge(match, team_a, team_b, raw_point_edge)
    point_edge += market_value_edge_points(value_a, value_b)
    if host_multiplier(team_a, match.venue) > 1.0:
        point_edge += 35.0
    if host_multiplier(team_b, match.venue) > 1.0:
        point_edge -= 35.0

    non_draw_a = 1.0 / (1.0 + math.exp(-point_edge / POINT_EDGE_SCALE))
    p_draw = adjusted_draw_probability(
        match,
        team_a,
        team_b,
        draw_probability(abs(rank_a.rank - rank_b.rank), profile_a, profile_b),
    )
    non_draw_mass = 1.0 - p_draw
    return non_draw_mass * non_draw_a, p_draw, non_draw_mass * (1.0 - non_draw_a)


def expected_goals(
    match: Match,
    rankings: dict[str, FifaRanking],
    profiles: dict[str, TeamProfile],
    baselines: ProfileBaselines,
    market_values: dict[str, MarketValue],
    club_cohesion: dict[str, ClubCohesion],
) -> tuple[float, float]:
    team_a = canonical_team(match.team_a)
    team_b = canonical_team(match.team_b)
    rank_a = rankings[team_a]
    rank_b = rankings[team_b]
    profile_a = profiles[team_a]
    profile_b = profiles[team_b]
    value_a = market_values[team_a]
    value_b = market_values[team_b]
    cohesion_a = club_cohesion[team_a]
    cohesion_b = club_cohesion[team_b]
    p_a, _, p_b = outcome_probabilities(match, rankings, profiles, market_values)
    style_point_edge = adjusted_point_edge(match, team_a, team_b, rank_a.points - rank_b.points)
    style_point_edge += market_value_edge_points(value_a, value_b)
    if host_multiplier(team_a, match.venue) > 1.0:
        style_point_edge += 35.0
    if host_multiplier(team_b, match.venue) > 1.0:
        style_point_edge -= 35.0
    style_effect = apply_style_influence_gate(
        style_matchup_effect(
            team_profile_features(profile_a),
            team_profile_features(profile_b),
        ),
        style_influence_factor(point_edge=style_point_edge),
    )

    total_goals = WORLD_CUP_GROUP_GOALS_PER_MATCH * style_total_modifier(profile_a, profile_b, baselines)
    total_goals *= style_effect.total_goal_multiplier
    total_goals *= ranking_tempo_multiplier(rank_a, rank_b)
    total_goals *= market_value_pair_tempo_multiplier(value_a, value_b)
    total_goals *= math.sqrt(cohesion_a.multiplier * cohesion_b.multiplier)
    total_goals *= in_tournament_tempo_multiplier(match, team_a, team_b)
    attack_a = profile_a.goals_for / max(0.2, profile_a.goals_for + profile_b.goals_for)
    defense_b = profile_b.goals_against / max(0.2, profile_a.goals_against + profile_b.goals_against)
    split_a = 0.50 + (p_a - p_b) * 0.55
    split_a += (attack_a - 0.5) * 0.30
    split_a += (defense_b - 0.5) * 0.22
    split_a += (profile_a.multi_goal_rate - profile_b.conceded_multi_rate) * 0.08
    split_a -= (profile_b.clean_sheet_rate - profile_a.clean_sheet_rate) * 0.06
    split_a += (rank_b.rank - rank_a.rank) / 400.0
    split_a += (
        math.log((value_a.total_eur_m + 1.0) / (value_b.total_eur_m + 1.0))
        * MARKET_VALUE_XG_SPLIT_WEIGHT
    )
    split_a += style_effect.xg_split_shift

    if host_multiplier(team_a, match.venue) > 1.0:
        split_a += 0.03
    if host_multiplier(team_b, match.venue) > 1.0:
        split_a -= 0.03

    split_a = clamp(split_a, 0.18, 0.82)
    lambda_a = total_goals * split_a
    lambda_b = total_goals * (1.0 - split_a)
    lambda_a *= ranking_attack_multiplier(rank_a, rank_b)
    lambda_b *= ranking_attack_multiplier(rank_b, rank_a)
    lambda_a *= market_value_attack_multiplier(value_a) * cohesion_a.multiplier
    lambda_b *= market_value_attack_multiplier(value_b) * cohesion_b.multiplier
    lambda_a, lambda_b = apply_open_favorite_underdog_xg(lambda_a, lambda_b, profile_a, profile_b, p_a, p_b)
    lambda_a, lambda_b = apply_in_tournament_attack_split_adjustment(match, team_a, team_b, lambda_a, lambda_b)
    return clamp(lambda_a, 0.05, 4.2), clamp(lambda_b, 0.05, 4.2)


def ranking_attack_multiplier(team: FifaRanking, opponent: FifaRanking) -> float:
    multiplier = 1.0
    if team.rank <= 20:
        multiplier *= TOP20_ATTACK_MULTIPLIER
    point_edge = max(0.0, team.points - opponent.points)
    rank_edge = max(0, opponent.rank - team.rank)
    if point_edge >= 120.0 or rank_edge >= 20:
        mismatch_boost = 1.0 + point_edge / MISMATCH_ATTACK_POINT_SCALE * 0.10 + rank_edge / 45.0 * 0.08
        multiplier *= min(MISMATCH_MAX_ATTACK_MULTIPLIER, mismatch_boost)
    return multiplier


def ranking_tempo_multiplier(rank_a: FifaRanking, rank_b: FifaRanking) -> float:
    multiplier = 1.0
    if (rank_a.rank <= 20) != (rank_b.rank <= 20):
        multiplier *= TOP20_VS_NON_TOP20_TEMPO_MULTIPLIER
    point_gap = abs(rank_a.points - rank_b.points)
    rank_gap = abs(rank_a.rank - rank_b.rank)
    if point_gap >= 120.0 or rank_gap >= 20:
        mismatch_boost = 1.0 + point_gap / MISMATCH_TEMPO_POINT_SCALE * 0.10 + rank_gap / 45.0 * 0.08
        multiplier *= min(MISMATCH_MAX_TEMPO_MULTIPLIER, mismatch_boost)
    return multiplier


def aggressive_score_lambdas(
    rank_a: int,
    rank_b: int,
    lambda_a: float,
    lambda_b: float,
) -> tuple[float, float]:
    tempo_multiplier = AGGRESSIVE_TOP20_VS_NON_TOP20_TEMPO_MULTIPLIER if (rank_a <= 20) != (rank_b <= 20) else 1.0
    attack_a = AGGRESSIVE_TOP20_ATTACK_MULTIPLIER if rank_a <= 20 else 1.0
    attack_b = AGGRESSIVE_TOP20_ATTACK_MULTIPLIER if rank_b <= 20 else 1.0
    return lambda_a * tempo_multiplier * attack_a, lambda_b * tempo_multiplier * attack_b


def in_tournament_signal(match: Match, team: str) -> float:
    if not in_tournament_adjustment_is_active(match, team):
        return 0.0
    return clamp(IN_TOURNAMENT_ADJUSTMENTS[team].points_adjustment / 180.0, -1.0, 1.0)


def adjusted_point_edge(match: Match, team_a: str, team_b: str, raw_point_edge: float) -> float:
    signal_a = in_tournament_signal(match, team_a)
    signal_b = in_tournament_signal(match, team_b)
    if signal_a == 0.0 and signal_b == 0.0:
        return raw_point_edge

    if raw_point_edge == 0.0:
        return (signal_a - signal_b) * IN_TOURNAMENT_EDGE_SHIFT_POINTS

    favorite_signal = signal_a if raw_point_edge > 0 else signal_b
    underdog_signal = signal_b if raw_point_edge > 0 else signal_a
    compression = clamp(
        (underdog_signal - favorite_signal) * IN_TOURNAMENT_EDGE_COMPRESSION_WEIGHT,
        -IN_TOURNAMENT_EDGE_COMPRESSION_LIMIT,
        IN_TOURNAMENT_EDGE_COMPRESSION_LIMIT,
    )
    adjusted_abs_edge = abs(raw_point_edge) * (1.0 - compression)
    adjusted_abs_edge = clamp(adjusted_abs_edge, 0.0, abs(raw_point_edge) * (1.0 + IN_TOURNAMENT_EDGE_COMPRESSION_LIMIT))
    adjusted = math.copysign(adjusted_abs_edge, raw_point_edge)
    adjusted += (signal_a - signal_b) * IN_TOURNAMENT_EDGE_SHIFT_POINTS
    return adjusted


def adjusted_draw_probability(match: Match, team_a: str, team_b: str, base_draw: float) -> float:
    signal_a = in_tournament_signal(match, team_a)
    signal_b = in_tournament_signal(match, team_b)
    if signal_a == 0.0 and signal_b == 0.0:
        return base_draw
    convergence = max(0.0, 1.0 - abs(signal_a - signal_b))
    both_under = max(0.0, -signal_a) + max(0.0, -signal_b)
    one_over_one_under = max(0.0, max(signal_a, signal_b) - min(signal_a, signal_b))
    draw_bonus = (
        convergence * both_under * IN_TOURNAMENT_DRAW_BOTH_UNDER_WEIGHT
        + min(one_over_one_under, 1.4) * IN_TOURNAMENT_DRAW_DIFF_WEIGHT
    )
    if in_tournament_goal_suppression(match, team_a, team_b) >= IN_TOURNAMENT_GOAL_SUPPRESSION_STRONG:
        draw_bonus += 0.025
    return clamp(base_draw + draw_bonus, 0.16, 0.42)


def in_tournament_goal_suppression(match: Match, team_a: str, team_b: str) -> float:
    if not in_tournament_adjustment_is_active(match, team_a) and not in_tournament_adjustment_is_active(match, team_b):
        return 0.0
    signal_a = in_tournament_signal(match, team_a)
    signal_b = in_tournament_signal(match, team_b)
    compression = abs(signal_a - signal_b) * 0.35
    both_under = (max(0.0, -signal_a) + max(0.0, -signal_b)) * 0.55
    one_under = max(0.0, -min(signal_a, signal_b)) * 0.45
    return clamp(compression + both_under + one_under, 0.0, 1.0)


def in_tournament_tempo_multiplier(match: Match, team_a: str, team_b: str) -> float:
    multiplier = 1.0
    active = False
    for team in (team_a, team_b):
        if in_tournament_adjustment_is_active(match, team):
            signal = in_tournament_signal(match, team)
            adjustment = IN_TOURNAMENT_ADJUSTMENTS[team]
            multiplier *= 1.0 + (adjustment.tempo_multiplier - 1.0) * IN_TOURNAMENT_TEMPO_STATE_WEIGHT
            multiplier *= 1.0 - max(0.0, -signal) * IN_TOURNAMENT_TEMPO_UNDER_WEIGHT
            active = True
    if not active:
        return 1.0
    return clamp(multiplier, 0.82, 1.18)


def apply_in_tournament_attack_split_adjustment(
    match: Match,
    team_a: str,
    team_b: str,
    lambda_a: float,
    lambda_b: float,
) -> tuple[float, float]:
    if not in_tournament_adjustment_is_active(match, team_a) and not in_tournament_adjustment_is_active(match, team_b):
        return lambda_a, lambda_b
    total = lambda_a + lambda_b
    if total <= 0:
        return lambda_a, lambda_b
    signal_a = in_tournament_signal(match, team_a)
    signal_b = in_tournament_signal(match, team_b)
    split_a = lambda_a / total
    split_a += (signal_a - signal_b) * IN_TOURNAMENT_SPLIT_SIGNAL_WEIGHT
    if in_tournament_adjustment_is_active(match, team_a):
        split_a += (IN_TOURNAMENT_ADJUSTMENTS[team_a].attack_multiplier - 1.0) * IN_TOURNAMENT_SPLIT_ATTACK_WEIGHT
    if in_tournament_adjustment_is_active(match, team_b):
        split_a -= (IN_TOURNAMENT_ADJUSTMENTS[team_b].attack_multiplier - 1.0) * IN_TOURNAMENT_SPLIT_ATTACK_WEIGHT
    split_a = clamp(split_a, 0.18, 0.82)
    return total * split_a, total * (1.0 - split_a)


def adjusted_points(match: Match, team: str, points: float) -> float:
    if not in_tournament_adjustment_is_active(match, team):
        return points
    return points + IN_TOURNAMENT_ADJUSTMENTS[team].points_adjustment


def market_value_base_goals(value_eur_m: float) -> int:
    capped = min(value_eur_m, 550.0)
    if capped < 50.0:
        return 0
    if capped < 150.0:
        return 1
    if capped < 260.0:
        return 2
    if capped < 390.0:
        return 3
    if capped < 520.0:
        return 4
    return 5


def clamp_score_goals(goals: int) -> int:
    return max(0, min(MAX_GOALS, goals))


def reduce_total_goals(goals_a: int, goals_b: int, max_total: int) -> tuple[int, int]:
    while goals_a + goals_b > max_total:
        if goals_a >= goals_b and goals_a > 0:
            goals_a -= 1
        elif goals_b > 0:
            goals_b -= 1
        else:
            break
    return goals_a, goals_b


def scale_score_to_top_goal(goals_a: int, goals_b: int, top_goal: int) -> tuple[int, int]:
    current_top = max(goals_a, goals_b)
    if current_top <= top_goal:
        return goals_a, goals_b
    scale = top_goal / current_top
    scaled_a = int(round(goals_a * scale))
    scaled_b = int(round(goals_b * scale))
    if goals_a > goals_b:
        scaled_a = top_goal
    elif goals_b > goals_a:
        scaled_b = top_goal
    else:
        scaled_a = top_goal
        scaled_b = top_goal
    return clamp_score_goals(scaled_a), clamp_score_goals(scaled_b)


def market_value_total_goals(
    value_a: MarketValue,
    value_b: MarketValue,
    ranking_a: FifaRanking,
    ranking_b: FifaRanking,
) -> int:
    total = market_value_base_goals(value_a.total_eur_m) + market_value_base_goals(value_b.total_eur_m)
    value_ratio_gap = abs(math.log((value_a.total_eur_m + 1.0) / (value_b.total_eur_m + 1.0)))
    rank_gap = abs(ranking_a.rank - ranking_b.rank)
    max_total = MARKET_VALUE_SCORE_MAX_TOTAL_GOALS
    if min(value_a.total_eur_m, value_b.total_eur_m) < 50.0 and (value_ratio_gap >= 2.0 or rank_gap >= 55):
        mismatch_bonus = round(max(0.0, value_ratio_gap - 1.5) * 1.6 + max(0.0, rank_gap - 50.0) / 15.0)
        total += min(5, mismatch_bonus)
        max_total = MARKET_VALUE_EXTREME_MISMATCH_MAX_TOTAL_GOALS
    return max(0, min(max_total, total))


def market_value_goal_share(
    value_a: MarketValue,
    value_b: MarketValue,
    ranking_a: FifaRanking,
    ranking_b: FifaRanking,
) -> float:
    value_edge = math.log((value_a.total_eur_m + 1.0) / (value_b.total_eur_m + 1.0))
    rank_edge = (ranking_b.rank - ranking_a.rank) / FIFA_RANK_SHARE_SCALE
    edge = value_edge * MARKET_VALUE_SHARE_WEIGHT + rank_edge * FIFA_RANK_SHARE_WEIGHT
    return 1.0 / (1.0 + math.exp(-edge))


def split_total_goals(total_goals: int, share_a: float) -> tuple[int, int]:
    goals_a = int(math.floor(total_goals * share_a + 0.5))
    goals_a = max(0, min(total_goals, goals_a))
    return goals_a, total_goals - goals_a


def market_value_score(
    value_a: MarketValue,
    value_b: MarketValue,
    ranking_a: FifaRanking,
    ranking_b: FifaRanking,
) -> tuple[int, int]:
    total_goals = market_value_total_goals(value_a, value_b, ranking_a, ranking_b)
    share_a = market_value_goal_share(value_a, value_b, ranking_a, ranking_b)
    raw_margin = total_goals * (share_a - 0.5) * 2.0
    if abs(raw_margin) < 0.5:
        outcome = "D"
    else:
        outcome = "A" if raw_margin > 0 else "B"
    signed_margin = legal_margin_for_total(total_goals, raw_margin, outcome)
    goals_a, goals_b = score_from_total_and_margin(total_goals, signed_margin)

    if value_a.total_eur_m >= 390.0 and value_b.total_eur_m >= 390.0:
        top_goal = 3 if value_a.total_eur_m >= 550.0 and value_b.total_eur_m >= 550.0 else 4
        goals_a, goals_b = scale_score_to_top_goal(goals_a, goals_b, top_goal)

    return reduce_total_goals(goals_a, goals_b, total_goals)


def compatible_total_goal_values(bucket: str, outcome: str) -> set[int]:
    values = total_goal_values_for_bucket(bucket)
    outcome = outcome_alias(outcome)
    if outcome == "D":
        return {total for total in values if total % 2 == 0}
    return {total for total in values if total > 0}


def choose_market_total_in_bucket(
    raw_total_goals: int,
    bucket: str,
    outcome: str,
    total_probabilities: dict[int, float] | None = None,
) -> int:
    candidates = sorted(compatible_total_goal_values(bucket, outcome) or total_goal_values_for_bucket(bucket))
    probabilities = total_probabilities or {}
    return min(
        candidates,
        key=lambda total: (
            abs(total - raw_total_goals),
            -probabilities.get(total, 0.0),
            total,
        ),
    )


def closest_total_distance_to_bucket(total_goals: int, bucket: str) -> int:
    return min(abs(total_goals - candidate) for candidate in total_goal_values_for_bucket(bucket))


def choose_market_bucket_from_options(
    raw_total_goals: int,
    bucket_options: set[str],
    total_probabilities: dict[int, float] | None = None,
) -> str:
    if not bucket_options:
        raise ValueError("bucket_options must not be empty")
    market_bucket = total_goal_bucket(raw_total_goals)
    if market_bucket in bucket_options:
        return market_bucket
    probabilities_by_bucket = {
        bucket: sum((total_probabilities or {}).get(total, 0.0) for total in total_goal_values_for_bucket(bucket))
        for bucket in bucket_options
    }
    return min(
        sorted(bucket_options),
        key=lambda bucket: (
            closest_total_distance_to_bucket(raw_total_goals, bucket),
            -probabilities_by_bucket[bucket],
        ),
    )


def market_value_score_in_bucket(
    value_a: MarketValue,
    value_b: MarketValue,
    ranking_a: FifaRanking,
    ranking_b: FifaRanking,
    bucket: str,
    total_probabilities: dict[int, float] | None = None,
) -> tuple[int, int]:
    raw_total_goals = market_value_total_goals(value_a, value_b, ranking_a, ranking_b)
    share_a = market_value_goal_share(value_a, value_b, ranking_a, ranking_b)
    raw_margin = raw_total_goals * (share_a - 0.5) * 2.0
    if abs(raw_margin) < 0.5:
        outcome = "D"
    else:
        outcome = "A" if raw_margin > 0 else "B"
    total_goals = choose_market_total_in_bucket(raw_total_goals, bucket, outcome, total_probabilities)
    bucket_margin = total_goals * (share_a - 0.5) * 2.0
    signed_margin = legal_margin_for_total(total_goals, bucket_margin, outcome)
    goals_a, goals_b = score_from_total_and_margin(total_goals, signed_margin)
    return reduce_total_goals(goals_a, goals_b, total_goals)


def market_value_score_in_buckets(
    value_a: MarketValue,
    value_b: MarketValue,
    ranking_a: FifaRanking,
    ranking_b: FifaRanking,
    bucket_options: set[str],
    total_probabilities: dict[int, float] | None = None,
) -> tuple[int, int]:
    raw_total_goals = market_value_total_goals(value_a, value_b, ranking_a, ranking_b)
    bucket = choose_market_bucket_from_options(raw_total_goals, bucket_options, total_probabilities)
    return market_value_score_in_bucket(
        value_a,
        value_b,
        ranking_a,
        ranking_b,
        bucket,
        total_probabilities,
    )


def market_value_promoted_total_goal_bucket(
    total_goal_buckets: list[tuple[str, float]],
    market_goals_a: int,
    market_goals_b: int,
) -> str:
    selected_bucket, selected_probability = total_goal_buckets[0]
    market_bucket = total_goal_bucket(market_goals_a + market_goals_b)
    if selected_bucket in HIGH_TOTAL_GOAL_BUCKETS:
        return selected_bucket
    if market_bucket not in HIGH_TOTAL_GOAL_BUCKETS:
        return selected_bucket
    high_candidates = [
        (bucket, probability)
        for bucket, probability in total_goal_buckets
        if bucket in HIGH_TOTAL_GOAL_BUCKETS
    ]
    if not high_candidates:
        return selected_bucket
    high_bucket, high_probability = high_candidates[0]
    if (
        high_probability >= selected_probability * MARKET_VALUE_HIGH_BUCKET_PROMOTION_RATIO
        and selected_probability <= MARKET_VALUE_HIGH_BUCKET_MAX_TOP_PROBABILITY
    ):
        return high_bucket
    return selected_bucket


def lower_total_goal_bucket(bucket: str) -> str:
    if bucket == "6-8球":
        return "4-5球"
    if bucket == "4-5球":
        return "2-3球"
    if bucket == "2-3球":
        return "0-1球"
    return bucket


def apply_in_tournament_total_goal_suppression(
    match: Match,
    team_a: str,
    team_b: str,
    selected_bucket: str,
    total_goal_buckets: list[tuple[str, float]],
    p_draw: float,
) -> str:
    suppression = in_tournament_goal_suppression(match, team_a, team_b)
    if suppression < IN_TOURNAMENT_GOAL_SUPPRESSION_MEDIUM:
        return selected_bucket

    probabilities = dict(total_goal_buckets)
    lower_bucket = lower_total_goal_bucket(selected_bucket)
    if lower_bucket == selected_bucket:
        return selected_bucket
    if selected_bucket == "2-3球":
        if suppression < IN_TOURNAMENT_LOW_BUCKET_SUPPRESSION or p_draw < IN_TOURNAMENT_LOW_BUCKET_DRAW:
            return selected_bucket

    selected_probability = probabilities.get(selected_bucket, 0.0)
    lower_probability = probabilities.get(lower_bucket, 0.0)
    threshold = 0.42 if suppression >= IN_TOURNAMENT_GOAL_SUPPRESSION_STRONG else 0.72
    if lower_probability >= selected_probability * threshold:
        return lower_bucket
    return selected_bucket


def choose_suppressed_second_total_goal_bucket(
    selected_bucket: str,
    total_goal_buckets: list[tuple[str, float]],
    suppression: float,
) -> str | None:
    if suppression < IN_TOURNAMENT_GOAL_SUPPRESSION_MEDIUM:
        return None
    lower_bucket = lower_total_goal_bucket(selected_bucket)
    if lower_bucket != selected_bucket:
        return lower_bucket
    return next((bucket for bucket, _ in total_goal_buckets if bucket != selected_bucket), None)


def choose_second_total_goal_bucket(
    total_goal_buckets: list[tuple[str, float]],
    selected_bucket: str,
    market_goals_a: int,
    market_goals_b: int,
    shape_labels: str = "",
) -> str:
    probabilities = dict(total_goal_buckets)
    default_bucket = next(bucket for bucket, _ in total_goal_buckets if bucket != selected_bucket)
    shape_set = {label for label in shape_labels.split(";") if label}

    if (
        selected_bucket == "4-5球"
        and (shape_set & {"open_game", "open_mismatch", "collapse_risk", "credible_opponent"})
        and probabilities.get("6-8球", 0.0) >= max(0.14, probabilities.get(default_bucket, 0.0) * 0.50)
    ):
        return "6-8球"

    if (
        selected_bucket == "2-3球"
        and (shape_set & {"low_block", "low_event_favorite", "low_event"})
        and probabilities.get("0-1球", 0.0) >= max(0.22, probabilities.get(default_bucket, 0.0) * 0.60)
    ):
        return "0-1球"

    return default_bucket


def format_total_goal_buckets(
    total_goal_buckets: list[tuple[str, float]],
    selected_bucket: str,
    second_bucket: str | None = None,
) -> str:
    priority = {selected_bucket: 0}
    if second_bucket is not None:
        priority[second_bucket] = 1
    ordered = sorted(
        total_goal_buckets,
        key=lambda item: (priority.get(item[0], 2), -item[1]),
    )
    return "; ".join(f"{bucket} {probability:.1%}" for bucket, probability in ordered)


def in_tournament_adjustment_is_active(match: Match, team: str) -> bool:
    if team not in IN_TOURNAMENT_ADJUSTMENTS:
        return False
    return match_datetime_bjt(match) > IN_TOURNAMENT_ADJUSTMENTS[team].effective_after_bjt


def match_datetime_bjt(match: Match) -> datetime:
    return datetime.combine(match.day_et, datetime.strptime(match.time_et, "%H:%M").time()) + timedelta(hours=12)


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def score_matrix(lambda_a: float, lambda_b: float) -> list[list[float]]:
    matrix: list[list[float]] = []
    total = 0.0
    for i in range(MAX_GOALS + 1):
        row: list[float] = []
        for j in range(MAX_GOALS + 1):
            p = poisson_pmf(i, lambda_a) * poisson_pmf(j, lambda_b)
            row.append(p)
            total += p
        matrix.append(row)
    if total <= 0:
        raise RuntimeError("score matrix has zero probability mass")
    return [[p / total for p in row] for row in matrix]


def outcome_adjusted_scores(
    lambda_a: float,
    lambda_b: float,
    p_a_target: float,
    p_draw_target: float,
    p_b_target: float,
) -> list[tuple[int, int, float]]:
    matrix = score_matrix(lambda_a, lambda_b)
    raw = {"A": 0.0, "D": 0.0, "B": 0.0}
    cells: list[tuple[int, int, str, float]] = []
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            outcome = "A" if i > j else "B" if i < j else "D"
            raw[outcome] += p
            cells.append((i, j, outcome, p))

    targets = {"A": p_a_target, "D": p_draw_target, "B": p_b_target}
    adjusted: list[tuple[int, int, float]] = []
    for i, j, outcome, p in cells:
        if raw[outcome] <= 0:
            raise RuntimeError(f"zero raw probability for outcome {outcome}")
        adjusted.append((i, j, p * targets[outcome] / raw[outcome]))
    total = sum(p for _, _, p in adjusted)
    if total <= 0:
        raise RuntimeError("adjusted score probabilities sum to zero")
    return sorted(((i, j, p / total) for i, j, p in adjusted), key=lambda cell: cell[2], reverse=True)


def top_total_goals(cells: list[tuple[int, int, float]]) -> list[tuple[int, float]]:
    totals: dict[int, float] = {}
    for goals_a, goals_b, probability in cells:
        total_goals = goals_a + goals_b
        totals[total_goals] = totals.get(total_goals, 0.0) + probability
    return sorted(totals.items(), key=lambda item: item[1], reverse=True)


def total_goal_bucket(total_goals: int) -> str:
    if total_goals <= 1:
        return "0-1球"
    if total_goals <= 3:
        return "2-3球"
    if total_goals <= 5:
        return "4-5球"
    return "6-8球"


def total_goal_bucket_from_expected(expected_total_goals: float) -> str:
    if expected_total_goals < 1.5:
        return "0-1球"
    if expected_total_goals < 3.5:
        return "2-3球"
    if expected_total_goals < 5.5:
        return "4-5球"
    return "6-8球"


def second_bucket_from_expected_total_goals(
    expected_total_goals: float,
    selected_bucket: str,
    shape_labels: str = "",
) -> str:
    labels = {label for label in shape_labels.split(";") if label}
    if selected_bucket == "0-1球":
        return "2-3球"
    if selected_bucket == "6-8球":
        return "4-5球"
    if selected_bucket == "2-3球":
        lower_distance = abs(expected_total_goals - 1.5)
        upper_distance = abs(3.5 - expected_total_goals)
        if labels & {"low_block", "low_event_favorite", "low_event"} and lower_distance <= upper_distance * 1.35:
            return "0-1球"
        if labels & {"open_game", "open_mismatch", "collapse_risk"} and upper_distance <= lower_distance * 1.35:
            return "4-5球"
        return "0-1球" if lower_distance < upper_distance else "4-5球"
    if selected_bucket == "4-5球":
        lower_distance = abs(expected_total_goals - 3.5)
        upper_distance = abs(5.5 - expected_total_goals)
        if labels & {"open_game", "open_mismatch", "collapse_risk", "credible_opponent"} and upper_distance <= lower_distance * 1.35:
            return "6-8球"
        if labels & {"low_block", "low_event_favorite", "low_event"} and lower_distance <= upper_distance * 1.35:
            return "2-3球"
        return "2-3球" if lower_distance < upper_distance else "6-8球"
    raise ValueError(f"unknown total goal bucket: {selected_bucket}")


def total_goal_values_for_bucket(bucket: str) -> set[int]:
    if bucket == "0-1球":
        return {0, 1}
    if bucket == "2-3球":
        return {2, 3}
    if bucket == "4-5球":
        return {4, 5}
    if bucket == "6-8球":
        return {6, 7, 8}
    raise ValueError(f"unknown total goal bucket: {bucket}")


def top_total_goal_buckets(total_goals: list[tuple[int, float]]) -> list[tuple[str, float]]:
    buckets = {label: 0.0 for label in TOTAL_GOAL_BUCKET_LABELS}
    for total, probability in total_goals:
        buckets[total_goal_bucket(total)] += probability
    return sorted(buckets.items(), key=lambda item: item[1], reverse=True)


def normalize_bucket_probabilities(probabilities: dict[str, float]) -> list[tuple[str, float]]:
    total = sum(probabilities.values())
    if total <= 0:
        raise RuntimeError("total-goal bucket probabilities sum to zero")
    return sorted(
        ((bucket, probability / total) for bucket, probability in probabilities.items()),
        key=lambda item: item[1],
        reverse=True,
    )


def multiply_bucket(probabilities: dict[str, float], bucket: str, factor: float) -> None:
    damped_factor = 1.0 + (factor - 1.0) * TOTAL_GOAL_BUCKET_ADJUSTMENT_STRENGTH
    probabilities[bucket] *= damped_factor


def total_goal_bucket_probabilities_from_expected(expected_total_goals: float) -> list[tuple[str, float]]:
    centers = {
        "0-1球": 0.8,
        "2-3球": 2.5,
        "4-5球": 4.5,
        "6-8球": 6.5,
    }
    sigma = 1.05
    probabilities = {
        bucket: math.exp(-((expected_total_goals - center) ** 2) / (2 * sigma * sigma))
        for bucket, center in centers.items()
    }
    return normalize_bucket_probabilities(probabilities)


def base_total_goal_bucket_probabilities_from_expected(expected_total_goals: float) -> list[tuple[str, float]]:
    probabilities = dict(total_goal_bucket_probabilities_from_expected(expected_total_goals))
    probabilities["2-3球"] *= BASE_TOTAL_GOAL_MID_BUCKET_FACTOR
    return normalize_bucket_probabilities(probabilities)


def base_total_goal_bucket_from_expected(expected_total_goals: float) -> str:
    selected_bucket = total_goal_bucket_from_expected(expected_total_goals)
    if selected_bucket != "2-3球":
        return selected_bucket
    return base_total_goal_bucket_probabilities_from_expected(expected_total_goals)[0][0]


def expected_total_goals_value(
    lambda_a: float,
    lambda_b: float,
    p_a: float,
    p_draw: float,
    p_b: float,
    ranking_a: FifaRanking | None = None,
    ranking_b: FifaRanking | None = None,
    profile_a: TeamProfile | None = None,
    profile_b: TeamProfile | None = None,
    baselines: ProfileBaselines | None = None,
    shape_labels: str = "",
) -> float:
    expected = lambda_a + lambda_b
    favorite_probability = max(p_a, p_b)
    outcome_edge = abs(p_a - p_b)
    expected += clamp((favorite_probability - 0.55) * 0.90, -0.18, 0.35)
    expected += clamp((outcome_edge - 0.35) * 0.35, -0.10, 0.22)
    expected -= clamp((p_draw - 0.29) * 1.05, -0.10, 0.28)

    if ranking_a is not None and ranking_b is not None:
        rank_gap = abs(ranking_a.rank - ranking_b.rank)
        point_gap = abs(ranking_a.points - ranking_b.points)
        expected += clamp((rank_gap - 18.0) / 95.0, -0.08, 0.18)
        expected += clamp((point_gap - 200.0) / 1600.0, -0.06, 0.16)

    if profile_a is not None and profile_b is not None and baselines is not None:
        style_set = {profile_a.style, profile_b.style}
        low_event_count = sum(style in LOW_EVENT_STYLES for style in style_set)
        high_event_count = sum(style in HIGH_EVENT_STYLES for style in style_set)
        expected -= low_event_count * 0.10
        expected += high_event_count * 0.10
        high_total_signal = (
            (profile_a.high_total_goal_rate - baselines.high_total_goal_rate)
            + (profile_b.high_total_goal_rate - baselines.high_total_goal_rate)
        ) / 2
        both_score_signal = (
            (profile_a.both_score_rate - baselines.both_score_rate)
            + (profile_b.both_score_rate - baselines.both_score_rate)
        ) / 2
        expected += clamp(high_total_signal * 0.55 + both_score_signal * 0.30, -0.22, 0.28)

    labels = {label for label in shape_labels.split(";") if label}
    is_low_event_shape = bool(labels & {"low_block", "low_event_favorite", "low_event"})
    is_controlled_favorite_shape = "controlled_favorite" in labels
    if labels & {"low_block", "low_event_favorite"}:
        expected -= 0.28
    if "low_event" in labels:
        expected -= 0.14
    if "transition_dog" in labels:
        expected += 0.03 if is_controlled_favorite_shape else 0.08
    if "set_piece_risk" in labels:
        expected += 0.05
    if "open_game" in labels:
        expected += 0.05 if is_controlled_favorite_shape else 0.28
    if is_controlled_favorite_shape:
        expected -= 0.75
    if "open_mismatch" in labels:
        expected += 0.42
    if "collapse_risk" in labels:
        expected += 0.65
    if expected >= 3.0 and not is_low_event_shape and not is_controlled_favorite_shape:
        expected += 0.25
    if (is_low_event_shape or is_controlled_favorite_shape) and p_draw >= 0.30:
        expected -= 0.20

    return clamp(expected, 0.4, 7.2)


def total_goal_bucket_probabilities(
    ranking_a: FifaRanking,
    ranking_b: FifaRanking,
    profile_a: TeamProfile,
    profile_b: TeamProfile,
    baselines: ProfileBaselines,
    value_a: MarketValue,
    value_b: MarketValue,
    lambda_a: float,
    lambda_b: float,
    p_a: float,
    p_draw: float,
    p_b: float,
    shape_labels: str = "",
    group_tempo_multiplier: float = 1.0,
) -> list[tuple[str, float]]:
    probabilities = dict(BASE_TOTAL_GOAL_BUCKET_PROBABILITIES)

    total_xg = lambda_a + lambda_b
    if total_xg < 2.05:
        multiply_bucket(probabilities, "0-1球", 1.24)
        multiply_bucket(probabilities, "4-5球", 0.82)
        multiply_bucket(probabilities, "6-8球", 0.72)
    elif total_xg > 3.15:
        multiply_bucket(probabilities, "0-1球", 0.62)
        multiply_bucket(probabilities, "4-5球", 1.26)
        multiply_bucket(probabilities, "6-8球", 1.46)
    elif total_xg > 2.75:
        multiply_bucket(probabilities, "0-1球", 0.80)
        multiply_bucket(probabilities, "4-5球", 1.15)
        multiply_bucket(probabilities, "6-8球", 1.18)

    rank_gap = abs(ranking_a.rank - ranking_b.rank)
    point_gap = abs(ranking_a.points - ranking_b.points)
    if rank_gap >= 35 or point_gap >= 180:
        multiply_bucket(probabilities, "0-1球", 0.86)
        multiply_bucket(probabilities, "4-5球", 1.18)
        multiply_bucket(probabilities, "6-8球", 1.22)
    elif rank_gap <= 10 and p_draw >= 0.28:
        multiply_bucket(probabilities, "0-1球", 1.12)
        multiply_bucket(probabilities, "6-8球", 0.84)

    value_gap = abs(math.log((value_a.total_eur_m + 1.0) / (value_b.total_eur_m + 1.0)))
    average_value = (value_a.total_eur_m + value_b.total_eur_m) / 2
    if value_gap >= 2.0:
        multiply_bucket(probabilities, "4-5球", 1.15)
        multiply_bucket(probabilities, "6-8球", 1.20)
    if average_value >= 390.0:
        multiply_bucket(probabilities, "4-5球", 1.12)
        multiply_bucket(probabilities, "0-1球", 0.88)

    style_set = {profile_a.style, profile_b.style}
    if style_set <= LOW_EVENT_STYLES:
        multiply_bucket(probabilities, "0-1球", 1.24)
        multiply_bucket(probabilities, "4-5球", 0.80)
        multiply_bucket(probabilities, "6-8球", 0.72)
    else:
        low_event_count = sum(style in LOW_EVENT_STYLES for style in style_set)
        high_event_count = sum(style in HIGH_EVENT_STYLES for style in style_set)
        for _ in range(low_event_count):
            multiply_bucket(probabilities, "0-1球", 1.10)
            multiply_bucket(probabilities, "6-8球", 0.90)
        for _ in range(high_event_count):
            multiply_bucket(probabilities, "0-1球", 0.88)
            multiply_bucket(probabilities, "4-5球", 1.13)
            multiply_bucket(probabilities, "6-8球", 1.10)

    high_total_signal = (
        (profile_a.high_total_goal_rate - baselines.high_total_goal_rate)
        + (profile_b.high_total_goal_rate - baselines.high_total_goal_rate)
    ) / 2
    both_score_signal = (
        (profile_a.both_score_rate - baselines.both_score_rate)
        + (profile_b.both_score_rate - baselines.both_score_rate)
    ) / 2
    high_factor = clamp(1.0 + high_total_signal * 0.65 + both_score_signal * 0.35, 0.78, 1.30)
    low_factor = clamp(1.0 - high_total_signal * 0.45 - both_score_signal * 0.20, 0.76, 1.24)
    multiply_bucket(probabilities, "0-1球", low_factor)
    multiply_bucket(probabilities, "4-5球", high_factor)
    multiply_bucket(probabilities, "6-8球", clamp(high_factor * 0.96, 0.78, 1.26))

    labels = {label for label in shape_labels.split(";") if label}
    if labels & LOW_TOTAL_GOAL_SHAPES:
        multiply_bucket(probabilities, "0-1球", 1.28)
        multiply_bucket(probabilities, "4-5球", 0.78)
        multiply_bucket(probabilities, "6-8球", 0.68)
    if "transition_dog" in labels or "set_piece_risk" in labels:
        multiply_bucket(probabilities, "2-3球", 1.06)
        multiply_bucket(probabilities, "4-5球", 1.10)
        multiply_bucket(probabilities, "0-1球", 0.90)
    if "open_game" in labels or "open_mismatch" in labels:
        multiply_bucket(probabilities, "0-1球", 0.72)
        multiply_bucket(probabilities, "4-5球", 1.26)
        multiply_bucket(probabilities, "6-8球", 1.26)
    if labels & EXTREME_TOTAL_GOAL_SHAPES:
        multiply_bucket(probabilities, "0-1球", 0.58)
        multiply_bucket(probabilities, "4-5球", 1.22)
        multiply_bucket(probabilities, "6-8球", 1.55)

    if group_tempo_multiplier < 0.96:
        multiply_bucket(probabilities, "0-1球", 1.12)
        multiply_bucket(probabilities, "4-5球", 0.92)
        multiply_bucket(probabilities, "6-8球", 0.86)
    elif group_tempo_multiplier > 1.04:
        multiply_bucket(probabilities, "0-1球", 0.90)
        multiply_bucket(probabilities, "4-5球", 1.10)
        multiply_bucket(probabilities, "6-8球", 1.10)

    if p_draw >= 0.32:
        multiply_bucket(probabilities, "0-1球", 1.12)
        multiply_bucket(probabilities, "6-8球", 0.82)
    if max(p_a, p_b) >= 0.70 and min(p_a, p_b) <= 0.10:
        multiply_bucket(probabilities, "0-1球", 0.82)
        multiply_bucket(probabilities, "4-5球", 1.12)
        multiply_bucket(probabilities, "6-8球", 1.18)

    return normalize_bucket_probabilities(probabilities)


def total_goal_probability_lookup(cells: list[tuple[int, int, float]]) -> dict[int, float]:
    return dict(top_total_goals(cells))


def score_outcome(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "A"
    if goals_b > goals_a:
        return "B"
    return "D"


def outcome_alias(value: str) -> str:
    if value in {"A", "home"}:
        return "A"
    if value in {"B", "away"}:
        return "B"
    if value in {"D", "draw"}:
        return "D"
    raise ValueError(f"unknown outcome: {value}")


def legal_margin_for_total(total_goals: int, raw_margin: float, outcome: str) -> int:
    outcome = outcome_alias(outcome)
    if outcome == "D":
        return 0
    sign = 1 if outcome == "A" else -1
    minimum_margin = 1 if total_goals % 2 else 2
    margins = list(range(minimum_margin, total_goals + 1, 2))
    if not margins:
        margins = [1]
    target = max(minimum_margin, min(total_goals, abs(raw_margin)))
    margin = min(margins, key=lambda item: (abs(item - target), item))
    return sign * margin


def adjust_margin_for_underdog_goal(
    total_goals: int,
    signed_margin: int,
    outcome: str,
    p_a: float,
    p_b: float,
    lambda_a: float,
    lambda_b: float,
) -> int:
    outcome = outcome_alias(outcome)
    if outcome == "D" or total_goals < UNDERDOG_GOAL_TOTAL_THRESHOLD:
        return signed_margin
    favorite_probability = p_a if outcome == "A" else p_b
    underdog_probability = p_b if outcome == "A" else p_a
    favorite_xg = lambda_a if outcome == "A" else lambda_b
    underdog_xg = lambda_b if outcome == "A" else lambda_a
    xg_edge = (favorite_xg - underdog_xg) / max(0.20, favorite_xg + underdog_xg)
    if favorite_probability > UNDERDOG_GOAL_MAX_WIN_PROBABILITY and xg_edge > UNDERDOG_GOAL_MAX_XG_EDGE:
        return signed_margin
    if underdog_xg < UNDERDOG_GOAL_MIN_XG and underdog_probability < UNDERDOG_GOAL_MIN_OUTCOME_PROBABILITY:
        return signed_margin

    sign = 1 if outcome == "A" else -1
    margin = abs(signed_margin)
    max_margin_with_underdog_goal = total_goals - 2
    if margin <= max_margin_with_underdog_goal:
        return signed_margin

    minimum_margin = 1 if total_goals % 2 else 2
    adjusted_margin = max(minimum_margin, max_margin_with_underdog_goal)
    if adjusted_margin % 2 != total_goals % 2:
        adjusted_margin = max(minimum_margin, adjusted_margin - 1)
    return sign * adjusted_margin


def score_from_total_and_margin(total_goals: int, signed_margin: int) -> tuple[int, int]:
    goals_a = (total_goals + signed_margin) // 2
    goals_b = total_goals - goals_a
    return goals_a, goals_b


def probability_margin(
    total_goals: int,
    predicted_outcome: str,
    p_a: float,
    p_b: float,
    lambda_a: float,
    lambda_b: float,
) -> float:
    outcome = outcome_alias(predicted_outcome)
    if outcome == "D":
        return 0.0
    if outcome == "A":
        probability_edge = (p_a - p_b) / max(0.20, p_a + p_b)
        xg_edge = (lambda_a - lambda_b) / max(0.20, lambda_a + lambda_b)
        return total_goals * (probability_edge * 0.65 + xg_edge * 0.35)
    probability_edge = (p_b - p_a) / max(0.20, p_a + p_b)
    xg_edge = (lambda_b - lambda_a) / max(0.20, lambda_a + lambda_b)
    return -total_goals * (probability_edge * 0.65 + xg_edge * 0.35)


def choose_total_goal_in_bucket(
    cells: list[tuple[int, int, float]],
    bucket: str,
    predicted_outcome: str,
    p_a: float,
    p_b: float,
    lambda_a: float,
    lambda_b: float,
) -> int:
    outcome = outcome_alias(predicted_outcome)
    probabilities = total_goal_probability_lookup(cells)
    candidates = sorted(total_goal_values_for_bucket(bucket))
    valid: list[tuple[int, float, float]] = []
    for total in candidates:
        if probabilities.get(total, 0.0) <= 0:
            continue
        raw_margin = probability_margin(total, outcome, p_a, p_b, lambda_a, lambda_b)
        signed_margin = legal_margin_for_total(total, raw_margin, outcome)
        signed_margin = adjust_margin_for_underdog_goal(total, signed_margin, outcome, p_a, p_b, lambda_a, lambda_b)
        goals_a, goals_b = score_from_total_and_margin(total, signed_margin)
        if score_outcome(goals_a, goals_b) == outcome:
            valid.append((total, probabilities[total], abs(raw_margin - signed_margin)))
    if not valid:
        return max(candidates, key=lambda total: probabilities.get(total, 0.0))
    best_probability = max(probability for _, probability, _ in valid)
    close = [item for item in valid if item[1] >= best_probability * 0.85]
    return min(close, key=lambda item: (item[2], -item[1], item[0]))[0]


def select_score_by_total_and_margin(
    cells: list[tuple[int, int, float]],
    total_goal_values: set[int],
    predicted_outcome: str,
    p_a: float,
    p_draw: float,
    p_b: float,
    lambda_a: float,
    lambda_b: float,
    excluded_scores: set[tuple[int, int]] | None = None,
) -> tuple[int, int, float]:
    del p_draw
    excluded_scores = excluded_scores or set()
    outcome = outcome_alias(predicted_outcome)
    candidates = [
        cell
        for cell in cells
        if (cell[0], cell[1]) not in excluded_scores
        and cell[0] + cell[1] in total_goal_values
        and score_outcome(cell[0], cell[1]) == outcome
    ]
    if not candidates:
        raise RuntimeError(f"no score found for total goals={sorted(total_goal_values)}, outcome={predicted_outcome}")

    total_probabilities: dict[int, float] = {}
    for goals_a, goals_b, probability in candidates:
        total_probabilities[goals_a + goals_b] = total_probabilities.get(goals_a + goals_b, 0.0) + probability
    target_total = max(total_probabilities.items(), key=lambda item: item[1])[0]
    raw_margin = probability_margin(target_total, outcome, p_a, p_b, lambda_a, lambda_b)
    target_margin = legal_margin_for_total(target_total, raw_margin, outcome)
    target_margin = adjust_margin_for_underdog_goal(
        target_total,
        target_margin,
        outcome,
        p_a,
        p_b,
        lambda_a,
        lambda_b,
    )
    target_score = score_from_total_and_margin(target_total, target_margin)
    for goals_a, goals_b, probability in candidates:
        if (goals_a, goals_b) == target_score:
            return goals_a, goals_b, probability
    return min(
        candidates,
        key=lambda cell: (
            abs((cell[0] + cell[1]) - target_total),
            abs((cell[0] - cell[1]) - target_margin),
            -cell[2],
        ),
    )


def best_score_for_total_goals(
    cells: list[tuple[int, int, float]],
    total_goal_values: set[int],
) -> tuple[int, int, float]:
    for goals_a, goals_b, probability in cells:
        if goals_a + goals_b in total_goal_values:
            return goals_a, goals_b, probability
    raise RuntimeError(f"no score found for total goals: {sorted(total_goal_values)}")


def best_score_outside_total_goal_buckets(
    cells: list[tuple[int, int, float]],
    complement_buckets: set[str],
) -> tuple[int, int, float]:
    for goals_a, goals_b, probability in cells:
        if total_goal_bucket(goals_a + goals_b) in complement_buckets:
            return goals_a, goals_b, probability
    raise RuntimeError(f"no score found in complement buckets: {sorted(complement_buckets)}")


def best_score_inside_total_goal_buckets(
    cells: list[tuple[int, int, float]],
    selected_buckets: set[str],
) -> tuple[int, int, float]:
    for goals_a, goals_b, probability in cells:
        if total_goal_bucket(goals_a + goals_b) in selected_buckets:
            return goals_a, goals_b, probability
    raise RuntimeError(f"no score found inside buckets: {sorted(selected_buckets)}")


def favorite_outcome_code(p_a: float, p_draw: float, p_b: float) -> str:
    return max(
        [("A", p_a), ("D", p_draw), ("B", p_b)],
        key=lambda item: item[1],
    )[0]


def outcome_probability(outcome: str, p_a: float, p_draw: float, p_b: float) -> float:
    if outcome == "A":
        return p_a
    if outcome == "D":
        return p_draw
    if outcome == "B":
        return p_b
    raise ValueError(f"unknown outcome: {outcome}")


def upset_score_outcomes(p_a: float, p_draw: float, p_b: float) -> tuple[set[str], bool]:
    favorite = favorite_outcome_code(p_a, p_draw, p_b)
    favorite_probability = outcome_probability(favorite, p_a, p_draw, p_b)
    if favorite == "D":
        return {"A", "B"}, False
    if favorite_probability >= UPSET_SCORE_STRONG_FAVORITE_PROBABILITY:
        return {favorite}, True
    underdog = "B" if favorite == "A" else "A"
    if favorite_probability >= UPSET_SCORE_MODERATE_FAVORITE_PROBABILITY:
        return {"D", underdog}, False
    return {"A", "D", "B"} - {favorite}, False


def select_upset_or_compression_score(
    cells: list[tuple[int, int, float]],
    selected_buckets: set[str],
    p_a: float,
    p_draw: float,
    p_b: float,
    excluded_scores: set[tuple[int, int]] | None = None,
) -> tuple[int, int, float]:
    excluded_scores = excluded_scores or set()
    outcomes, compression_mode = upset_score_outcomes(p_a, p_draw, p_b)
    candidates = [
        cell
        for cell in cells
        if (cell[0], cell[1]) not in excluded_scores
        and total_goal_bucket(cell[0] + cell[1]) in selected_buckets
        and score_outcome(cell[0], cell[1]) in outcomes
    ]
    if candidates and compression_mode:
        return min(
            candidates,
            key=lambda cell: (
                cell[0] + cell[1],
                abs(cell[0] - cell[1]),
                -cell[2],
            ),
        )
    if candidates:
        return candidates[0]

    fallback_candidates = [
        cell
        for cell in cells
        if (cell[0], cell[1]) not in excluded_scores
        and total_goal_bucket(cell[0] + cell[1]) in selected_buckets
    ]
    if fallback_candidates:
        return fallback_candidates[0]
    return best_score_inside_total_goal_buckets(cells, selected_buckets)


def score_matches_outcome(goals_a: int, goals_b: int, predicted_outcome: str) -> bool:
    if predicted_outcome in {"A", "home"}:
        return goals_a > goals_b
    if predicted_outcome in {"D", "draw"}:
        return goals_a == goals_b
    if predicted_outcome in {"B", "away"}:
        return goals_a < goals_b
    raise ValueError(f"unknown predicted outcome: {predicted_outcome}")


def select_recommended_score(
    cells: list[tuple[int, int, float]],
    predicted_outcome: str,
) -> tuple[tuple[int, int, float], list[tuple[int, int, float]], list[tuple[int, float]]]:
    aligned_scores = [
        cell
        for cell in cells
        if score_matches_outcome(cell[0], cell[1], predicted_outcome)
    ]
    if not aligned_scores:
        raise RuntimeError(f"no score cells aligned with outcome {predicted_outcome}")

    total_goals = top_total_goals(cells)
    total_goal_buckets = top_total_goal_buckets(total_goals)
    if total_goal_buckets[0][0] in HIGH_TOTAL_GOAL_BUCKETS and predicted_outcome in {"A", "B", "home", "away"}:
        high_bucket_candidates = [
            cell
            for cell in aligned_scores
            if total_goal_bucket(cell[0] + cell[1]) == total_goal_buckets[0][0]
        ]
        if high_bucket_candidates:
            return high_bucket_candidates[0], aligned_scores, total_goals
    for total, _ in total_goals:
        candidates = [
            cell
            for cell in aligned_scores
            if cell[0] + cell[1] == total
        ]
        if candidates:
            return candidates[0], aligned_scores, total_goals
    return aligned_scores[0], aligned_scores, total_goals


def predict_match(
    match: Match,
    rankings: dict[str, FifaRanking],
    profiles: dict[str, TeamProfile],
    baselines: ProfileBaselines,
    market_values: dict[str, MarketValue],
    club_cohesion: dict[str, ClubCohesion],
) -> dict:
    team_a = canonical_team(match.team_a)
    team_b = canonical_team(match.team_b)
    p_a, p_draw, p_b = outcome_probabilities(match, rankings, profiles, market_values)
    lambda_a, lambda_b = expected_goals(match, rankings, profiles, baselines, market_values, club_cohesion)
    cells = outcome_adjusted_scores(lambda_a, lambda_b, p_a, p_draw, p_b)
    predicted_outcome = predicted_outcome_from_probabilities(p_a, p_draw, p_b)
    total_goals = top_total_goals(cells)
    expected_total_goals = expected_total_goals_value(
        lambda_a,
        lambda_b,
        p_a,
        p_draw,
        p_b,
        rankings[team_a],
        rankings[team_b],
        profiles[team_a],
        profiles[team_b],
        baselines,
    )
    total_goal_buckets = base_total_goal_bucket_probabilities_from_expected(expected_total_goals)
    market_raw_goals_a, market_raw_goals_b = market_value_score(
        market_values[team_a],
        market_values[team_b],
        rankings[team_a],
        rankings[team_b],
    )
    selected_bucket = base_total_goal_bucket_from_expected(expected_total_goals)
    selected_bucket = apply_in_tournament_total_goal_suppression(
        match,
        team_a,
        team_b,
        selected_bucket,
        total_goal_buckets,
        p_draw,
    )
    suppression = in_tournament_goal_suppression(match, team_a, team_b)
    complement_bucket = choose_suppressed_second_total_goal_bucket(
        selected_bucket,
        total_goal_buckets,
        suppression,
    ) or second_bucket_from_expected_total_goals(expected_total_goals, selected_bucket)
    aggressive_lambda_a, aggressive_lambda_b = aggressive_score_lambdas(rankings[team_a].rank, rankings[team_b].rank, lambda_a, lambda_b)
    aggressive_cells = outcome_adjusted_scores(aggressive_lambda_a, aggressive_lambda_b, p_a, p_draw, p_b)
    recommended, aligned_scores, _ = select_recommended_score(cells, predicted_outcome)
    aggressive_recommended, aggressive_aligned_scores, _ = select_recommended_score(aggressive_cells, predicted_outcome)
    selected_total_goal_bucket_labels = {selected_bucket}
    complement_total_goal_bucket_labels = {complement_bucket}
    selected_total_goal = choose_total_goal_in_bucket(
        cells,
        selected_bucket,
        predicted_outcome,
        p_a,
        p_b,
        lambda_a,
        lambda_b,
    )
    complement_total_goal = choose_total_goal_in_bucket(
        aggressive_cells,
        complement_bucket,
        predicted_outcome,
        p_a,
        p_b,
        aggressive_lambda_a,
        aggressive_lambda_b,
    )
    selected_total_goal_values = {selected_total_goal}
    try:
        recommended = select_score_by_total_and_margin(
            cells,
            selected_total_goal_values,
            predicted_outcome,
            p_a,
            p_draw,
            p_b,
            lambda_a,
            lambda_b,
        )
    except RuntimeError:
        pass
    try:
        aggressive_recommended = select_score_by_total_and_margin(
            aggressive_cells,
            selected_total_goal_values,
            predicted_outcome,
            p_a,
            p_draw,
            p_b,
            aggressive_lambda_a,
            aggressive_lambda_b,
        )
    except RuntimeError:
        pass
    try:
        total_constrained_score = select_score_by_total_and_margin(
            cells,
            selected_total_goal_values,
            predicted_outcome,
            p_a,
            p_draw,
            p_b,
            lambda_a,
            lambda_b,
        )
    except RuntimeError:
        total_constrained_score = best_score_for_total_goals(cells, selected_total_goal_values)
    try:
        bucket_primary_score = select_score_by_total_and_margin(
            aggressive_aligned_scores,
            selected_total_goal_values,
            predicted_outcome,
            p_a,
            p_draw,
            p_b,
            aggressive_lambda_a,
            aggressive_lambda_b,
        )
    except RuntimeError:
        bucket_primary_score = best_score_inside_total_goal_buckets(aggressive_aligned_scores, selected_total_goal_bucket_labels)
    try:
        bucket_complement_score = select_score_by_total_and_margin(
            aggressive_aligned_scores,
            {complement_total_goal},
            predicted_outcome,
            p_a,
            p_draw,
            p_b,
            aggressive_lambda_a,
            aggressive_lambda_b,
            {(bucket_primary_score[0], bucket_primary_score[1])},
        )
    except RuntimeError:
        bucket_complement_score = best_score_outside_total_goal_buckets(aggressive_aligned_scores, complement_total_goal_bucket_labels)
    aggressive_recommended = bucket_complement_score
    market_goals_a, market_goals_b = market_value_score_in_buckets(
        market_values[team_a],
        market_values[team_b],
        rankings[team_a],
        rankings[team_b],
        {selected_bucket, complement_bucket},
        total_goal_probability_lookup(cells),
    )
    upset_recommended = select_upset_or_compression_score(
        cells,
        {selected_bucket, complement_bucket},
        p_a,
        p_draw,
        p_b,
        {
            (bucket_primary_score[0], bucket_primary_score[1]),
            (bucket_complement_score[0], bucket_complement_score[1]),
            (market_goals_a, market_goals_b),
        },
    )
    rank_gap = abs(rankings[team_a].rank - rankings[team_b].rank)
    uncertainty, top_probability, outcome_margin = outcome_uncertainty(p_a, p_draw, p_b)
    bjt = match_datetime_bjt(match)
    style_features_a = team_profile_features(profiles[team_a])
    style_features_b = team_profile_features(profiles[team_b])
    style_point_edge = adjusted_point_edge(match, team_a, team_b, rankings[team_a].points - rankings[team_b].points)
    style_point_edge += market_value_edge_points(market_values[team_a], market_values[team_b])
    if host_multiplier(team_a, match.venue) > 1.0:
        style_point_edge += 35.0
    if host_multiplier(team_b, match.venue) > 1.0:
        style_point_edge -= 35.0
    style_influence = style_influence_factor(point_edge=style_point_edge)
    style_effect = apply_style_influence_gate(
        style_matchup_effect(style_features_a, style_features_b),
        style_influence,
    )
    return {
        "group": match.group,
        "date_bjt": bjt.strftime("%Y-%m-%d"),
        "time_bjt": bjt.strftime("%H:%M"),
        "date_et": match.day_et.isoformat(),
        "time_et": match.time_et,
        "team_a": team_a,
        "team_b": team_b,
        "venue": match.venue,
        "fifa_rank_a": rankings[team_a].rank,
        "fifa_rank_b": rankings[team_b].rank,
        "fifa_points_a": adjusted_points(match, team_a, rankings[team_a].points),
        "fifa_points_b": adjusted_points(match, team_b, rankings[team_b].points),
        "style_a": profiles[team_a].style,
        "style_b": profiles[team_b].style,
        "style_features_a": ";".join(style_features_a),
        "style_features_b": ";".join(style_features_b),
        "style_matchup_edge": style_effect.edge,
        "style_matchup_influence": style_influence,
        "style_matchup_points_shift": style_effect.points_shift,
        "style_matchup_total_multiplier": style_effect.total_goal_multiplier,
        "style_matchup_reasons": "; ".join(style_effect.reasons),
        "xg_a": lambda_a,
        "xg_b": lambda_b,
        "aggressive_xg_a": aggressive_lambda_a,
        "aggressive_xg_b": aggressive_lambda_b,
        "market_value_a_eur_m": market_values[team_a].total_eur_m,
        "market_value_b_eur_m": market_values[team_b].total_eur_m,
        "club_cohesion_a": club_cohesion[team_a].multiplier,
        "club_cohesion_b": club_cohesion[team_b].multiplier,
        "top_club_a": club_cohesion[team_a].top_club,
        "top_club_b": club_cohesion[team_b].top_club,
        "top_club_players_a": club_cohesion[team_a].top_club_players,
        "top_club_players_b": club_cohesion[team_b].top_club_players,
        "score": f"{recommended[0]}-{recommended[1]}",
        "score_probability": recommended[2],
        "predicted_outcome": predicted_outcome,
        "recommended_score": f"{recommended[0]}-{recommended[1]}",
        "recommended_score_probability": recommended[2],
        "aggressive_score": f"{aggressive_recommended[0]}-{aggressive_recommended[1]}",
        "aggressive_score_probability": aggressive_recommended[2],
        "mode_total_goals": total_goals[0][0],
        "selected_total_goals": f"{selected_total_goal}/{complement_total_goal}",
        "selected_total_goal_bucket": selected_bucket,
        "total_constrained_score": f"{total_constrained_score[0]}-{total_constrained_score[1]}",
        "total_constrained_score_probability": total_constrained_score[2],
        "bucket_primary_score": f"{bucket_primary_score[0]}-{bucket_primary_score[1]}",
        "bucket_primary_score_probability": bucket_primary_score[2],
        "bucket_complement_score": f"{bucket_complement_score[0]}-{bucket_complement_score[1]}",
        "bucket_complement_score_probability": bucket_complement_score[2],
        "market_value_raw_score": f"{market_raw_goals_a}-{market_raw_goals_b}",
        "market_value_score": f"{market_goals_a}-{market_goals_b}",
        "upset_score": f"{upset_recommended[0]}-{upset_recommended[1]}",
        "upset_score_probability": upset_recommended[2],
        "top_outcome_probability": top_probability,
        "outcome_margin": outcome_margin,
        "uncertainty_score": uncertainty,
        "risk_label": risk_label(uncertainty),
        "risk_reasons": risk_reasons(p_a, p_draw, p_b, rank_gap, total_goals[0][0]),
        "p_a": p_a,
        "p_draw": p_draw,
        "p_b": p_b,
        "top_total_goals": "; ".join(f"{total}球 {p:.1%}" for total, p in total_goals[:3]),
        "top_total_goal_buckets": format_total_goal_buckets(total_goal_buckets, selected_bucket, complement_bucket),
        "top_scores": "; ".join(f"{i}-{j} {p:.1%}" for i, j, p in aligned_scores[:3]),
        "top_aggressive_scores": "; ".join(f"{i}-{j} {p:.1%}" for i, j, p in aggressive_aligned_scores[:3]),
    }


def write_outputs(predictions: list[dict], rankings: dict[str, FifaRanking]) -> None:
    fieldnames = [
        "group",
        "date_bjt",
        "time_bjt",
        "date_et",
        "time_et",
        "team_a",
        "team_b",
        "venue",
        "fifa_rank_a",
        "fifa_rank_b",
        "fifa_points_a",
        "fifa_points_b",
        "style_a",
        "style_b",
        "style_features_a",
        "style_features_b",
        "style_matchup_edge",
        "style_matchup_influence",
        "style_matchup_points_shift",
        "style_matchup_total_multiplier",
        "style_matchup_reasons",
        "xg_a",
        "xg_b",
        "aggressive_xg_a",
        "aggressive_xg_b",
        "market_value_a_eur_m",
        "market_value_b_eur_m",
        "club_cohesion_a",
        "club_cohesion_b",
        "top_club_a",
        "top_club_b",
        "top_club_players_a",
        "top_club_players_b",
        "score",
        "score_probability",
        "predicted_outcome",
        "recommended_score",
        "recommended_score_probability",
        "aggressive_score",
        "aggressive_score_probability",
        "mode_total_goals",
        "selected_total_goals",
        "selected_total_goal_bucket",
        "total_constrained_score",
        "total_constrained_score_probability",
        "bucket_primary_score",
        "bucket_primary_score_probability",
        "bucket_complement_score",
        "bucket_complement_score_probability",
        "market_value_raw_score",
        "market_value_score",
        "upset_score",
        "upset_score_probability",
        "top_outcome_probability",
        "outcome_margin",
        "uncertainty_score",
        "risk_label",
        "risk_reasons",
        "p_a",
        "p_draw",
        "p_b",
        "top_total_goals",
        "top_total_goal_buckets",
        "top_scores",
        "top_aggressive_scores",
    ]
    with PREDICTIONS_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    snapshot_date = sorted({ranking.snapshot_date for ranking in rankings.values()})[0]
    lines = [
        "# 2026 世界杯小组赛预测：球队画像模型",
        "",
        f"- FIFA 排名日期：{snapshot_date}",
        "- 球队画像三大属性：实时强化 FIFA 排名、10 年滚动历史风格、Transfermarkt 当前球员身价。",
        "- 配合度来自 26 人名单的俱乐部集中度，只做小幅补正。",
        "- 预测生成不读取未来赛果。",
        "",
        "| 北京时间 | 组 | 比赛 | 风格 | 胜负参考 | 模型 | 备选 | 身价 | 爆冷 | 风险 | 胜/平/负 | xG | 备选xG | 身价 | 配合度 | 总进球桶 | 总进球候选 | 模型前三比分 | 备选前三比分 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---|",
    ]
    for row in predictions:
        outcome_text = {
            "A": f"{row['team_a']}胜",
            "D": "平",
            "B": f"{row['team_b']}胜",
        }[row["predicted_outcome"]]
        lines.append(
            "| {date} {time} | {group} | {a}({ra}) vs {b}({rb}) | {sa}/{sb} | {outcome} | "
            "{model_score} ({model_p:.1%}) | {aggressive_score} ({aggressive_p:.1%}) | {mv_score} | {upset_score} ({upset_p:.1%}) | "
            "{risk} | {pa:.1%}/{pd:.1%}/{pb:.1%} | "
                "{xa:.2f}-{xb:.2f} | {axa:.2f}-{axb:.2f} | €{mva:.1f}m/€{mvb:.1f}m | "
            "{ca:.2f}({tca}:{tcpa})/{cb:.2f}({tcb}:{tcpb}) | {buckets} | {totals} | {tops} | {aggressive_tops} |".format(
                date=row["date_bjt"],
                time=row["time_bjt"],
                group=row["group"],
                a=row["team_a"],
                b=row["team_b"],
                ra=row["fifa_rank_a"],
                rb=row["fifa_rank_b"],
                sa=row["style_a"],
                sb=row["style_b"],
                outcome=outcome_text,
                model_score=row["recommended_score"],
                model_p=row["recommended_score_probability"],
                aggressive_score=row["aggressive_score"],
                aggressive_p=row["aggressive_score_probability"],
                mv_score=row["market_value_score"],
                upset_score=row["upset_score"],
                upset_p=row["upset_score_probability"],
                risk=f"{row['risk_label']}：{row['risk_reasons']}",
                pa=row["p_a"],
                pd=row["p_draw"],
                pb=row["p_b"],
                xa=row["xg_a"],
                xb=row["xg_b"],
                axa=row["aggressive_xg_a"],
                axb=row["aggressive_xg_b"],
                mva=row["market_value_a_eur_m"],
                mvb=row["market_value_b_eur_m"],
                ca=row["club_cohesion_a"],
                cb=row["club_cohesion_b"],
                tca=row["top_club_a"],
                tcb=row["top_club_b"],
                tcpa=row["top_club_players_a"],
                tcpb=row["top_club_players_b"],
                buckets=row["top_total_goal_buckets"],
                totals=row["top_total_goals"],
                tops=row["top_scores"],
                aggressive_tops=row["top_aggressive_scores"],
            )
        )
    PREDICTIONS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    matches = schedule()
    rankings = load_fifa_rankings()
    profiles = load_profiles()
    market_values = load_market_values()
    club_cohesion = load_club_cohesion()
    assert_inputs_cover_schedule(rankings, profiles, market_values, club_cohesion, matches)
    baselines = profile_baselines(list(profiles.values()))
    predictions = [
        predict_match(match, rankings, profiles, baselines, market_values, club_cohesion)
        for match in matches
    ]
    write_outputs(predictions, rankings)

    print(f"Wrote {PREDICTIONS_CSV}")
    print(f"Wrote {PREDICTIONS_MD}")
    for row in predictions[:12]:
        print(
            f"{row['date_bjt']} {row['time_bjt']} {row['team_a']} vs {row['team_b']}: "
            f"{row['recommended_score']} {row['p_a']:.1%}/{row['p_draw']:.1%}/{row['p_b']:.1%}"
        )


if __name__ == "__main__":
    main()
