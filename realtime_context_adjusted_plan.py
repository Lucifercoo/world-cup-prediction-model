from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from predict import DATA_DIR, OUTPUT_DIR, canonical_team, schedule
from predict_fifa_profile import (
    MAX_GOALS,
    PREDICTIONS_CSV,
    TOTAL_GOAL_BUCKET_LABELS,
    adjust_margin_for_underdog_goal,
    aggressive_score_lambdas,
    choose_market_bucket_from_options,
    choose_second_total_goal_bucket,
    choose_total_goal_in_bucket,
    expected_total_goals_value,
    format_total_goal_buckets,
    load_market_values,
    match_datetime_bjt,
    outcome_adjusted_scores,
    poisson_pmf,
    select_recommended_score,
    select_score_by_total_and_margin,
    total_goal_bucket,
    total_goal_bucket_probabilities_from_expected,
    total_goal_probability_lookup,
)
from predict_fifa_profile import (
    second_bucket_from_expected_total_goals as profile_second_bucket_from_expected_total_goals,
)
from reports import realtime_output
from style_matchups import (
    StyleMatchupEffect,
    apply_style_influence_gate,
    prediction_row_style_features,
    style_influence_factor,
    style_matchup_effect,
    team_shape_style_features,
)

CONTEXT_CSV = DATA_DIR / "realtime_team_context.csv"
MATCH_SHAPE_CSV = DATA_DIR / "match_shape_context.csv"
TEAM_SHAPE_PROFILE_CSV = DATA_DIR / "in_tournament_team_shape_profiles.csv"
KEY_PLAYER_SIGNAL_CSV = DATA_DIR / "world_cup_2026_key_player_signals.csv"
KEY_PLAYER_MATCH_STATUS_CSV = DATA_DIR / "world_cup_2026_key_player_match_status.csv"
RESULTS_CSV = DATA_DIR / "world_cup_2026_results.csv"
INTERNATIONAL_RESULTS_CSV = DATA_DIR / "international_results.csv"
ADJUSTED_CSV = OUTPUT_DIR / "realtime_context_adjusted_plan.csv"
ADJUSTED_MD = OUTPUT_DIR / "realtime_context_adjusted_plan.md"
REALTIME_CACHE_DIR = OUTPUT_DIR / "realtime_context_cache"
BJT = timezone(timedelta(hours=8))
HIGH_CONTEXT_TEMPO_DISCOUNT = 0.92
STRONG_CONTEXT_TEMPO_DISCOUNT = 0.84
MARKET_VALUE_MAX_TOTAL = 6
HIGH_TOTAL_GOAL_BUCKETS = {"4-5球", "6-8球"}
ATTACK_CONTEXT_MIN_MULTIPLIER = 0.56
ATTACK_CONTEXT_MAX_MULTIPLIER = 1.38
ATTACK_CONTEXT_WEIGHTS = {
    "home_adaptation_multiplier": 1.50,
    "travel_multiplier": 3.00,
    "weather_multiplier": 1.50,
    "cohesion_multiplier": 2.00,
    "injury_multiplier": 8.00,
}
DEFENSE_LEAK_CONTEXT_MIN_MULTIPLIER = 0.88
DEFENSE_LEAK_CONTEXT_MAX_MULTIPLIER = 1.14
TEMPO_CONTEXT_MIN_MULTIPLIER = 0.88
TEMPO_CONTEXT_MAX_MULTIPLIER = 1.12
TEAM_CONTEXT_TOTAL_BUDGET = 0.30
DEFENSE_LEAK_EVIDENCE_WORDS = (
    "defensive injury",
    "key defender",
    "defender",
    "defensive concern",
    "defensive risk",
    "defensive confidence",
    "goalkeeper",
    "keeper",
    "centre-back",
    "center-back",
    "back line",
    "backline",
    "conceded",
    "exposed defensive",
    "defensive/creative injury",
    "cesar montes",
    "kounde",
    "bensebaini",
    "ndicka",
    "aguerd",
    "mazraoui",
    "neuer",
)
UNDERDOG_GOAL_EVIDENCE_WORDS = (
    "transition threat",
    "counter",
    "set piece threat",
    "set-piece threat",
    "set piece route",
    "set-piece route",
    "set pieces and",
    "set-pieces and",
    "direct attack",
    "direct attacks",
    "scoring route",
    "attacking life",
    "front line",
    "pace",
    "high line",
    "low block",
    "showed transition quality",
)
STRONG_FAVORITE_PROBABILITY = 0.58
MAX_DRAW_PROBABILITY = 0.46
MIN_DRAW_PROBABILITY = 0.14
THREE_WAY_CLOSE_MARGIN = 0.08
DRAW_CLOSE_TO_TOP_MARGIN = 0.15
XG_DRAW_BASE_MARGIN = 0.10
XG_DRAW_FIRST_ROUND_BOTH_NEUTRAL_FACTOR = 1.80
XG_DRAW_ACCEPTABLE_FACTOR = 1.35
XG_DRAW_MUST_CHASE_FACTOR = 0.70
XG_DRAW_LOW_EVENT_FACTOR = 1.40
XG_DRAW_OPEN_GAME_FACTOR = 0.70
XG_DRAW_LOW_TEMPO_FACTOR = 1.15
XG_DRAW_HIGH_TEMPO_FACTOR = 0.85
XG_OUTCOME_PROBABILITY_CONFLICT = 0.18
KNOCKOUT_DRAW_CLOSE_MARGIN = 0.10
KNOCKOUT_LOW_EVENT_DRAW_MARGIN = 0.31
KNOCKOUT_LOW_EVENT_MIN_DRAW_PROBABILITY = 0.28
KNOCKOUT_LOW_EVENT_DRAW_LABELS = {"low_block", "low_event", "low_event_favorite"}
KNOCKOUT_DRAW_XG_VETO_MARGIN = 0.30
MODEL_SCORE_DRAW_AGAINST_WIN_MAX_EDGE = 0.15
MODEL_SCORE_DRAW_AGAINST_WIN_MIN_DRAW_PROBABILITY = 0.25
KNOCKOUT_HIGH_BUCKET_CAP_MIN_PROBABILITY = 0.66
KNOCKOUT_HIGH_BUCKET_CAP_MIN_FAVORITE_XG = 3.00
KNOCKOUT_HIGH_BUCKET_CAP_MIN_XG_GAP = 2.50
KNOCKOUT_HIGH_BUCKET_BACKUP_FACTOR = 0.45
KNOCKOUT_HIGH_BUCKET_TAIL_FACTOR = 0.35
KNOCKOUT_HIGH_BUCKET_LOW_EVENT_DRAW_MIN_PROBABILITY = 0.28
KNOCKOUT_EARLY_HIGH_BUCKET_CAP_STAGES = {"R32", "R16"}
KNOCKOUT_LATE_HIGH_BUCKET_CAP_STAGES = {"QF", "SF", "FINAL"}
KNOCKOUT_TRUE_OPEN_LABELS = {"open_game", "open_mismatch", "collapse_risk"}
KNOCKOUT_HIGH_BUCKET_EXTREME_EXEMPT_LABELS = {"open_mismatch", "collapse_risk"}
KNOCKOUT_EARLY_CONTROL_CAP_MIN_PROBABILITY = 0.45
KNOCKOUT_EARLY_CONTROL_CAP_MIN_FAVORITE_XG = 2.20
KNOCKOUT_EARLY_CONTROL_CAP_MIN_XG_GAP = 1.00
KNOCKOUT_EARLY_CONTROL_CAP_MAX_UNDERDOG_XG = 1.35
KNOCKOUT_SCORE_LADDER_CLOSE_XG_GAP = 0.55
KNOCKOUT_SCORE_LADDER_CLOSE_DRAW_MIN_PROBABILITY = 0.25
KNOCKOUT_SCORE_LADDER_HIGH_DRAW_MIN_PROBABILITY = 0.28
KNOCKOUT_SCORE_LADDER_STRONG_FAVORITE_XG = 2.75
KNOCKOUT_SCORE_LADDER_STRONG_UNDERDOG_XG_MAX = 0.90
KNOCKOUT_SCORE_LADDER_STRONG_XG_GAP = 1.70
KNOCKOUT_SCORE_LADDER_STRONG_DRAW_MAX_PROBABILITY = 0.27
KNOCKOUT_LOW_EVENT_HIGH_BUCKET_LABELS = {"low_block", "low_event", "low_event_favorite", "controlled_favorite"}
GROUP_STAGE_DRAW_MIN_MULTIPLIER = 0.82
GROUP_STAGE_DRAW_MAX_MULTIPLIER = 1.34
GROUP_STAGE_TEMPO_MIN_MULTIPLIER = 0.88
GROUP_STAGE_TEMPO_MAX_MULTIPLIER = 1.14
GROUP_STAGE_QUALIFIED_ATTACK_MULTIPLIER = 0.94
GROUP_STAGE_QUALIFIED_TEMPO_MULTIPLIER = 0.97
GROUP_STAGE_QUALIFIED_TEMPO_WITH_CHASER_MULTIPLIER = 0.98
GROUP_STAGE_BOTH_QUALIFIED_TEMPO_MULTIPLIER = 0.90
GROUP_STAGE_QUALIFIED_DEFENSE_LEAK_MULTIPLIER = 1.10
GROUP_STAGE_QUALIFIED_CHASER_ATTACK_MULTIPLIER = 1.08
GROUP_STAGE_QUALIFIED_OPEN_TAIL_TOTAL_GOALS = 0.28
GROUP_KNOCKOUT_ROUTE_PRESSURE = {
    "I": {"runner_up_opponent": "Argentina", "pressure": 0.28},
}
QUALIFIED_LOW_BUCKET_DRAW_MIN_PROBABILITY = 0.30
QUALIFIED_LOW_BUCKET_DRAW_MAX_EDGE = 0.12
QUALIFIED_LOW_BUCKET_WIN_DISCOUNT = 0.80
QUALIFIED_LOW_BUCKET_DRAW_BOOST = 1.35
QUALIFIED_UPSET_RISK_MAX_CHASER_DRAW_ACCEPTANCE = 0.82
GROUP_DRAW_SLOWDOWN_ACCEPTANCE_START = 0.95
GROUP_DRAW_SLOWDOWN_ACCEPTANCE_FULL = 1.16
GROUP_DRAW_SLOWDOWN_MIN_ACCEPTANCE_START = 0.80
GROUP_DRAW_SLOWDOWN_MIN_ACCEPTANCE_FULL = 1.12
GROUP_DRAW_SLOWDOWN_DRAW_START = 0.22
GROUP_DRAW_SLOWDOWN_DRAW_FULL = 0.32
GROUP_DRAW_SLOWDOWN_MAX_DROP = 0.42
GROUP_DRAW_SLOWDOWN_ROUND_ONE_FACTOR = 0.0
HIGH_DRAW_ACCEPTANCE_TEMPO_WEIGHT = 0.28
PREMATCH_SHAPE_DRAW_MIN_MULTIPLIER = 0.90
PREMATCH_SHAPE_DRAW_MAX_MULTIPLIER = 1.25
PREMATCH_SHAPE_TEMPO_MIN_MULTIPLIER = 0.88
PREMATCH_SHAPE_TEMPO_MAX_MULTIPLIER = 1.12
PREMATCH_SHAPE_FAVORITE_ATTACK_MIN_MULTIPLIER = 0.82
PREMATCH_SHAPE_FAVORITE_ATTACK_MAX_MULTIPLIER = 1.12
PREMATCH_SHAPE_UNDERDOG_ATTACK_MIN_MULTIPLIER = 0.90
PREMATCH_SHAPE_UNDERDOG_ATTACK_MAX_MULTIPLIER = 1.18
STYLE_EDGE_LABEL_A = "style_edge_a"
STYLE_EDGE_LABEL_B = "style_edge_b"
STYLE_EDGE_FAVORED_PROBABILITY_MULTIPLIER = 1.22
STYLE_EDGE_SUPPRESSED_PROBABILITY_MULTIPLIER = 0.88
STYLE_EDGE_DRAW_PROBABILITY_MULTIPLIER = 0.94
STYLE_EDGE_FAVORED_XG_MULTIPLIER = 1.08
STYLE_EDGE_SUPPRESSED_XG_MULTIPLIER = 0.82
STYLE_EDGE_DRAW_MARGIN_FACTOR = 0.60
LOW_BLOCK_LOW_BUCKET_MIN_EDGE = 0.08
CREDIBLE_OPPONENT_MAX_BUCKET = "4-5球"
RECENT_GOAL_LOOKBACK_MATCHES = 5
RECENT_BIG_GOAL_THRESHOLD = 4
REALTIME_TOTAL_GOAL_BUCKET_ADJUSTMENT_STRENGTH = 0.50
MARKET_SCORE_MARGIN_DISCOUNT = 0.75
AGGRESSIVE_SCORE_CONFIDENCE_THRESHOLD = 1.02
SOURCE_CONFIDENCE_MIN_MULTIPLIER = 0.92
SOURCE_CONFIDENCE_MAX_MULTIPLIER = 1.10
LINEUP_CERTAINTY_MIN_MULTIPLIER = 0.90
LINEUP_CERTAINTY_MAX_MULTIPLIER = 1.05
STRONG_FAVORITE_LOW_BUCKET_MIN_FAVORITE_XG = float(
    os.environ.get("WC_STRONG_FAVORITE_LOW_BUCKET_MIN_FAVORITE_XG", "2.20")
)
STRONG_FAVORITE_LOW_BUCKET_MAX_UNDERDOG_XG = float(
    os.environ.get("WC_STRONG_FAVORITE_LOW_BUCKET_MAX_UNDERDOG_XG", "0.50")
)
STRONG_FAVORITE_LOW_BUCKET_MIN_PROBABILITY = float(
    os.environ.get("WC_STRONG_FAVORITE_LOW_BUCKET_MIN_PROBABILITY", str(STRONG_FAVORITE_PROBABILITY))
)
HIGH_VALUE_SOURCE_LABELS = {"opta", "odds", "market", "betting", "lineup"}
WEATHER_SOURCE_LABELS = {"weather", "local_weather"}
LINEUP_UNCERTAINTY_WORDS = (
    "questionable",
    "major doubt",
    "game-time",
    "day-to-day",
    "not expected",
    "unlikely",
    "bench",
    "managed from the bench",
    "experimental",
    "debutants",
    "suspended",
    "ruled out",
    "out for",
    "hamstring",
    "calf",
    "injury concerns",
    "fitness notes",
)
LINEUP_IMPORTANT_PLAYER_WORDS = (
    "key defender",
    "core",
    "salah",
    "neymar",
    "pulisic",
    "yildiz",
    "valencia",
    "yamal",
    "nico williams",
    "dzeko",
    "montes",
    "sithole",
    "zwane",
    "ndicka",
    "mitoma",
    "endo",
)
LINEUP_STABILITY_WORDS = (
    "available",
    "fit and projected",
    "projected to start",
    "expected to start",
    "no major injury",
    "no injury concerns",
    "no strong injury",
    "close to full strength",
    "full strength",
)
STAR_LIMITING_WORDS = (
    "not expected",
    "unlikely",
    "bench",
    "managed from the bench",
    "doubt",
    "questionable",
    "hamstring",
    "calf",
    "not pushed upward",
    "not expected for a full 90",
    "real doubt",
)
STAR_STARTING_WORDS = (
    "projected to start",
    "expected to start",
    "fit and projected",
    "projected to play",
    "close to full strength",
)
STAR_VALUE_SHARE_REFERENCE = 0.18
STAR_VALUE_SHARE_MIN_SCALE = 0.45
STAR_VALUE_SHARE_MAX_SCALE = 1.75
STAR_RANK_GOAL_SCALE_MIN = 0.55
STAR_RANK_GOAL_SCALE_MAX = 1.30
STAR_RANK_GOAL_SCALE_SPAN = 42.0
STAR_GOAL_BONUS_MIN = -0.22
STAR_GOAL_BONUS_MAX = 0.28
TEAM_SHAPE_PROFILE_MODE = os.environ.get("WC_TEAM_SHAPE_PROFILE_MODE", "off").strip().lower()
TEAM_SHAPE_PROFILE_MICRO_STRENGTH = 0.35


@dataclass(frozen=True)
class TeamContext:
    home_adaptation_multiplier: float
    travel_multiplier: float
    weather_multiplier: float
    cohesion_multiplier: float
    injury_multiplier: float
    attack_multiplier: float
    opponent_attack_multiplier: float
    tempo_multiplier: float
    source_confidence_multiplier: float
    lineup_certainty_multiplier: float
    notes: str
    source_urls: str
    weather_high_c: str
    travel_km: str
    defense_leak_evidence: bool
    underdog_goal_evidence: bool


@dataclass(frozen=True)
class KeyPlayerSignal:
    team: str
    key_player: str
    aliases: tuple[str, ...]
    market_value_eur_m: float
    override_value_share: float | None
    star_type: str
    goal_bonus: float
    notes: str


@dataclass(frozen=True)
class KeyPlayerMatchStatus:
    status: str
    impact: float
    source_urls: str
    notes: str


@dataclass(frozen=True)
class KeyPlayerEffect:
    goal_bonus: float
    label: str


@dataclass(frozen=True)
class MatchShapeContext:
    pre_match_shapes: str
    observed_shapes: str
    draw_multiplier: float
    tempo_multiplier: float
    favorite_attack_multiplier: float
    underdog_attack_multiplier: float
    notes: str


@dataclass(frozen=True)
class TeamShapeProfile:
    effective_after_bjt: datetime
    played: int
    strong_defense_attack_suppression_score: float
    draw_multiplier: float
    tempo_multiplier: float
    attack_multiplier: float
    opponent_attack_multiplier: float
    derived_labels: frozenset[str]
    reason: str


@dataclass(frozen=True)
class GroupStanding:
    played: int = 0
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0

    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against


@dataclass(frozen=True)
class CompletedMatch:
    kickoff_bjt: datetime
    group: str
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int


@dataclass(frozen=True)
class ScoredMatch:
    kickoff_bjt: datetime
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int


@dataclass(frozen=True)
class RecentGoalSignal:
    team: str
    totals: tuple[int, ...]

    @property
    def match_count(self) -> int:
        return len(self.totals)

    @property
    def big_goal_count(self) -> int:
        return sum(1 for total in self.totals if total >= RECENT_BIG_GOAL_THRESHOLD)

    @property
    def has_big_goal(self) -> bool:
        return self.big_goal_count > 0


@dataclass(frozen=True)
class GroupStageContext:
    round_number: int
    draw_acceptance_a: float
    draw_acceptance_b: float
    draw_multiplier: float
    tempo_multiplier: float
    attack_multiplier_a: float
    attack_multiplier_b: float
    opponent_attack_multiplier_a: float
    opponent_attack_multiplier_b: float
    open_tail_total_goals: float
    qualified_a: bool
    qualified_b: bool
    complete: bool
    notes: str


def is_knockout_stage(group: str) -> bool:
    return group.upper() in {"R32", "R16", "QF", "SF", "FINAL", "3P"}


def as_float(value: str, default: float = 1.0) -> float:
    text = value.strip()
    if text == "":
        return default
    return float(text)


def clamp_value(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def contains_any_alias(text: str, aliases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(alias in lowered for alias in aliases)


def sentence_window_for_alias(text: str, aliases: tuple[str, ...]) -> str:
    lowered = text.lower()
    sentences = re.split(r"[.;|]", lowered)
    matching = [sentence for sentence in sentences if any(alias in sentence for alias in aliases)]
    return " ".join(matching)


def source_label_set(source_urls: str) -> set[str]:
    labels: set[str] = set()
    for part in source_urls.split(";"):
        piece = part.strip()
        if ":" not in piece:
            continue
        labels.add(piece.split(":", 1)[0].strip().lower())
    return labels


def source_confidence_multiplier(source_urls: str, notes: str) -> float:
    labels = source_label_set(source_urls) - WEATHER_SOURCE_LABELS
    multiplier = 1.0
    if len(labels) >= 3:
        multiplier += 0.06
    elif len(labels) == 2:
        multiplier += 0.03
    elif len(labels) == 0:
        multiplier -= 0.04
    if labels & HIGH_VALUE_SOURCE_LABELS:
        multiplier += 0.02
    lowered = notes.lower()
    if "repeatedly" in lowered or "both rate" in lowered or "also notes" in lowered:
        multiplier += 0.02
    if "fixed-source-search" in labels:
        multiplier -= 0.03
    return clamp_value(
        multiplier,
        SOURCE_CONFIDENCE_MIN_MULTIPLIER,
        SOURCE_CONFIDENCE_MAX_MULTIPLIER,
    )


def lineup_certainty_multiplier(notes: str, source_urls: str) -> float:
    lowered = f"{notes} {source_urls}".lower()
    multiplier = 1.0
    if contains_any_keyword(lowered, LINEUP_UNCERTAINTY_WORDS):
        multiplier -= 0.04
    if contains_any_keyword(lowered, LINEUP_IMPORTANT_PLAYER_WORDS):
        multiplier -= 0.02
    if contains_any_keyword(lowered, LINEUP_STABILITY_WORDS):
        multiplier += 0.02
    return clamp_value(
        multiplier,
        LINEUP_CERTAINTY_MIN_MULTIPLIER,
        LINEUP_CERTAINTY_MAX_MULTIPLIER,
    )


def normalize_context_deltas(
    attack_delta: float,
    defense_leak_delta: float,
    tempo_delta: float,
) -> tuple[float, float, float]:
    total = abs(attack_delta) + abs(defense_leak_delta) + abs(tempo_delta)
    if total <= TEAM_CONTEXT_TOTAL_BUDGET or total == 0:
        return attack_delta, defense_leak_delta, tempo_delta
    scale = TEAM_CONTEXT_TOTAL_BUDGET / total
    return attack_delta * scale, defense_leak_delta * scale, tempo_delta * scale


def team_context_multipliers(
    home_adaptation_multiplier: float,
    travel_multiplier: float,
    weather_multiplier: float,
    cohesion_multiplier: float,
    injury_multiplier: float,
    raw_defense_leak_multiplier: float,
    raw_tempo_multiplier: float,
) -> tuple[float, float, float]:
    attack_delta = (
        (home_adaptation_multiplier - 1.0) * ATTACK_CONTEXT_WEIGHTS["home_adaptation_multiplier"]
        + (travel_multiplier - 1.0) * ATTACK_CONTEXT_WEIGHTS["travel_multiplier"]
        + (weather_multiplier - 1.0) * ATTACK_CONTEXT_WEIGHTS["weather_multiplier"]
        + (cohesion_multiplier - 1.0) * ATTACK_CONTEXT_WEIGHTS["cohesion_multiplier"]
        + (injury_multiplier - 1.0) * ATTACK_CONTEXT_WEIGHTS["injury_multiplier"]
    )
    defense_leak_delta = raw_defense_leak_multiplier - 1.0
    tempo_delta = raw_tempo_multiplier - 1.0
    attack_delta, defense_leak_delta, tempo_delta = normalize_context_deltas(
        attack_delta,
        defense_leak_delta,
        tempo_delta,
    )
    return (
        clamp_value(1.0 + attack_delta, ATTACK_CONTEXT_MIN_MULTIPLIER, ATTACK_CONTEXT_MAX_MULTIPLIER),
        clamp_value(
            1.0 + defense_leak_delta,
            DEFENSE_LEAK_CONTEXT_MIN_MULTIPLIER,
            DEFENSE_LEAK_CONTEXT_MAX_MULTIPLIER,
        ),
        clamp_value(1.0 + tempo_delta, TEMPO_CONTEXT_MIN_MULTIPLIER, TEMPO_CONTEXT_MAX_MULTIPLIER),
    )


def load_context() -> dict[tuple[str, str], TeamContext]:
    contexts: dict[tuple[str, str], TeamContext] = {}
    with CONTEXT_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            home_adaptation_multiplier = as_float(row["home_adaptation_multiplier"])
            travel_multiplier = as_float(row["travel_multiplier"])
            weather_multiplier = as_float(row["weather_multiplier"])
            cohesion_multiplier = as_float(row["cohesion_multiplier"])
            injury_multiplier = as_float(row["injury_multiplier"])
            raw_defense_leak_multiplier = as_float(row["opponent_attack_multiplier"])
            raw_tempo_multiplier = as_float(row["tempo_multiplier"])
            notes = row["analysis_notes"].strip()
            attack_multiplier, defense_leak_multiplier, tempo_multiplier = team_context_multipliers(
                home_adaptation_multiplier,
                travel_multiplier,
                weather_multiplier,
                cohesion_multiplier,
                injury_multiplier,
                raw_defense_leak_multiplier,
                raw_tempo_multiplier,
            )
            source_confidence = source_confidence_multiplier(row["source_urls"].strip(), notes)
            lineup_certainty = lineup_certainty_multiplier(notes, row["source_urls"].strip())
            contexts[(row["match"], row["team"])] = TeamContext(
                home_adaptation_multiplier=home_adaptation_multiplier,
                travel_multiplier=travel_multiplier,
                weather_multiplier=weather_multiplier,
                cohesion_multiplier=cohesion_multiplier,
                injury_multiplier=injury_multiplier,
                attack_multiplier=attack_multiplier,
                opponent_attack_multiplier=defense_leak_multiplier,
                tempo_multiplier=tempo_multiplier,
                source_confidence_multiplier=source_confidence,
                lineup_certainty_multiplier=lineup_certainty,
                notes=notes,
                source_urls=row["source_urls"].strip(),
                weather_high_c=row["weather_high_c"].strip(),
                travel_km=row["travel_km"].strip(),
                defense_leak_evidence=contains_any_keyword(notes, DEFENSE_LEAK_EVIDENCE_WORDS),
                underdog_goal_evidence=contains_any_keyword(notes, UNDERDOG_GOAL_EVIDENCE_WORDS),
            )
    return contexts


def load_key_player_signals() -> dict[str, KeyPlayerSignal]:
    if not KEY_PLAYER_SIGNAL_CSV.exists():
        raise RuntimeError(
            f"missing required user-supplied key-player signal input: {KEY_PLAYER_SIGNAL_CSV}. "
            "See docs/DATA_FETCH.csv and docs/DATA_SOURCES.md for acquisition requirements."
        )
    signals: dict[str, KeyPlayerSignal] = {}
    with KEY_PLAYER_SIGNAL_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            team = canonical_team(row["team"])
            aliases = tuple(alias.strip().lower() for alias in row["aliases"].split("|") if alias.strip())
            signals[team] = KeyPlayerSignal(
                team=team,
                key_player=row["key_player"].strip(),
                aliases=aliases,
                market_value_eur_m=float(row["market_value_eur_m"]),
                override_value_share=(
                    float(row["override_value_share"])
                    if row.get("override_value_share", "").strip()
                    else None
                ),
                star_type=row["star_type"].strip(),
                goal_bonus=float(row["goal_bonus"]),
                notes=row["notes"].strip(),
            )
    return signals


def load_key_player_match_statuses() -> dict[tuple[str, str, str], KeyPlayerMatchStatus]:
    if not KEY_PLAYER_MATCH_STATUS_CSV.exists():
        return {}
    statuses: dict[tuple[str, str, str], KeyPlayerMatchStatus] = {}
    with KEY_PLAYER_MATCH_STATUS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            statuses[
                (
                    f"{row['date_bjt'].strip()} {row['time_bjt'].strip()}",
                    row["match"].strip(),
                    canonical_team(row["team"]),
                )
            ] = KeyPlayerMatchStatus(
                status=row["status"].strip(),
                impact=float(row["impact"]),
                source_urls=row["source_urls"].strip(),
                notes=row["notes"].strip(),
            )
    return statuses


def load_match_shapes() -> dict[str, MatchShapeContext]:
    if not MATCH_SHAPE_CSV.exists():
        return {}
    shapes: dict[str, MatchShapeContext] = {}
    with MATCH_SHAPE_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            pre_match_shapes = row["pre_match_shapes"].strip()
            if not pre_match_shapes:
                continue
            shapes[row["match"]] = MatchShapeContext(
                pre_match_shapes=pre_match_shapes,
                observed_shapes=row["observed_shapes"].strip(),
                draw_multiplier=clamp_value(
                    as_float(row["draw_multiplier"]),
                    PREMATCH_SHAPE_DRAW_MIN_MULTIPLIER,
                    PREMATCH_SHAPE_DRAW_MAX_MULTIPLIER,
                ),
                tempo_multiplier=clamp_value(
                    as_float(row["tempo_multiplier"]),
                    PREMATCH_SHAPE_TEMPO_MIN_MULTIPLIER,
                    PREMATCH_SHAPE_TEMPO_MAX_MULTIPLIER,
                ),
                favorite_attack_multiplier=clamp_value(
                    as_float(row["favorite_attack_multiplier"]),
                    PREMATCH_SHAPE_FAVORITE_ATTACK_MIN_MULTIPLIER,
                    PREMATCH_SHAPE_FAVORITE_ATTACK_MAX_MULTIPLIER,
                ),
                underdog_attack_multiplier=clamp_value(
                    as_float(row["underdog_attack_multiplier"]),
                    PREMATCH_SHAPE_UNDERDOG_ATTACK_MIN_MULTIPLIER,
                    PREMATCH_SHAPE_UNDERDOG_ATTACK_MAX_MULTIPLIER,
                ),
                notes=row["notes"].strip(),
            )
    return shapes


def load_team_shape_profiles() -> dict[str, list[TeamShapeProfile]]:
    if not TEAM_SHAPE_PROFILE_CSV.exists():
        return {}
    profiles: dict[str, list[TeamShapeProfile]] = {}
    with TEAM_SHAPE_PROFILE_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            team = canonical_team(row["team"])
            profiles.setdefault(team, []).append(
                TeamShapeProfile(
                    effective_after_bjt=datetime.strptime(row["effective_after_bjt"], "%Y-%m-%d %H:%M"),
                    played=int(row["played"]),
                    strong_defense_attack_suppression_score=as_float(
                        row.get("strong_defense_attack_suppression_score", ""),
                        0.0,
                    ),
                    draw_multiplier=float(row["draw_multiplier"]),
                    tempo_multiplier=float(row["tempo_multiplier"]),
                    attack_multiplier=float(row["attack_multiplier"]),
                    opponent_attack_multiplier=float(row["opponent_attack_multiplier"]),
                    derived_labels=frozenset(label for label in row["derived_labels"].split(";") if label),
                    reason=row["reason"].strip(),
                )
            )
    for rows in profiles.values():
        rows.sort(key=lambda profile: profile.effective_after_bjt)
    return profiles


def team_shape_profile_at(
    profiles: dict[str, list[TeamShapeProfile]],
    team: str,
    kickoff_bjt: datetime,
) -> TeamShapeProfile | None:
    selected: TeamShapeProfile | None = None
    for profile in profiles.get(canonical_team(team), []):
        if profile.effective_after_bjt >= kickoff_bjt:
            break
        selected = profile
    return selected


def joined_shape_labels(labels: set[str]) -> str:
    preferred = [
        "low_event",
        "controlled_favorite",
        "strong_defense_attack_suppression",
        "transition_dog",
        "open_game",
        "collapse_risk",
    ]
    ordered = [label for label in preferred if label in labels]
    ordered.extend(sorted(label for label in labels if label not in set(ordered)))
    return ";".join(ordered)


def merge_shape_contexts(
    explicit_shape: MatchShapeContext | None,
    profile_shape: MatchShapeContext | None,
) -> MatchShapeContext | None:
    if explicit_shape is None:
        return profile_shape
    if profile_shape is None:
        return explicit_shape
    labels = set(explicit_shape.pre_match_shapes.split(";")) | set(profile_shape.pre_match_shapes.split(";"))
    labels.discard("")
    return MatchShapeContext(
        pre_match_shapes=joined_shape_labels(labels),
        observed_shapes=explicit_shape.observed_shapes,
        draw_multiplier=clamp_value(
            explicit_shape.draw_multiplier * profile_shape.draw_multiplier,
            PREMATCH_SHAPE_DRAW_MIN_MULTIPLIER,
            PREMATCH_SHAPE_DRAW_MAX_MULTIPLIER,
        ),
        tempo_multiplier=clamp_value(
            explicit_shape.tempo_multiplier * profile_shape.tempo_multiplier,
            PREMATCH_SHAPE_TEMPO_MIN_MULTIPLIER,
            PREMATCH_SHAPE_TEMPO_MAX_MULTIPLIER,
        ),
        favorite_attack_multiplier=clamp_value(
            explicit_shape.favorite_attack_multiplier * profile_shape.favorite_attack_multiplier,
            PREMATCH_SHAPE_FAVORITE_ATTACK_MIN_MULTIPLIER,
            PREMATCH_SHAPE_FAVORITE_ATTACK_MAX_MULTIPLIER,
        ),
        underdog_attack_multiplier=clamp_value(
            explicit_shape.underdog_attack_multiplier * profile_shape.underdog_attack_multiplier,
            PREMATCH_SHAPE_UNDERDOG_ATTACK_MIN_MULTIPLIER,
            PREMATCH_SHAPE_UNDERDOG_ATTACK_MAX_MULTIPLIER,
        ),
        notes=" | ".join(note for note in (explicit_shape.notes, profile_shape.notes) if note),
    )


def scale_shape_context(
    shape: MatchShapeContext,
    *,
    strength: float,
    keep_labels: bool,
) -> MatchShapeContext:
    return MatchShapeContext(
        pre_match_shapes=shape.pre_match_shapes if keep_labels else "",
        observed_shapes=shape.observed_shapes,
        draw_multiplier=clamp_value(
            1.0 + (shape.draw_multiplier - 1.0) * strength,
            PREMATCH_SHAPE_DRAW_MIN_MULTIPLIER,
            PREMATCH_SHAPE_DRAW_MAX_MULTIPLIER,
        ),
        tempo_multiplier=clamp_value(
            1.0 + (shape.tempo_multiplier - 1.0) * strength,
            PREMATCH_SHAPE_TEMPO_MIN_MULTIPLIER,
            PREMATCH_SHAPE_TEMPO_MAX_MULTIPLIER,
        ),
        favorite_attack_multiplier=clamp_value(
            1.0 + (shape.favorite_attack_multiplier - 1.0) * strength,
            PREMATCH_SHAPE_FAVORITE_ATTACK_MIN_MULTIPLIER,
            PREMATCH_SHAPE_FAVORITE_ATTACK_MAX_MULTIPLIER,
        ),
        underdog_attack_multiplier=clamp_value(
            1.0 + (shape.underdog_attack_multiplier - 1.0) * strength,
            PREMATCH_SHAPE_UNDERDOG_ATTACK_MIN_MULTIPLIER,
            PREMATCH_SHAPE_UNDERDOG_ATTACK_MAX_MULTIPLIER,
        ),
        notes=shape.notes,
    )


def select_shape_context(
    explicit_shape: MatchShapeContext | None,
    profile_shape: MatchShapeContext | None,
) -> MatchShapeContext | None:
    if TEAM_SHAPE_PROFILE_MODE == "off" or profile_shape is None:
        return explicit_shape
    if TEAM_SHAPE_PROFILE_MODE == "micro":
        micro_shape = scale_shape_context(
            profile_shape,
            strength=TEAM_SHAPE_PROFILE_MICRO_STRENGTH,
            keep_labels=False,
        )
        return merge_shape_contexts(explicit_shape, micro_shape)
    if TEAM_SHAPE_PROFILE_MODE == "labels":
        return merge_shape_contexts(explicit_shape, profile_shape)
    raise RuntimeError(f"unknown WC_TEAM_SHAPE_PROFILE_MODE: {TEAM_SHAPE_PROFILE_MODE}")


def inferred_team_shape_context(
    profile_a: TeamShapeProfile | None,
    profile_b: TeamShapeProfile | None,
    team_a: str,
    team_b: str,
    p_a: float,
    p_b: float,
) -> MatchShapeContext | None:
    if profile_a is None and profile_b is None:
        return None
    favorite_is_a = p_a >= p_b
    favorite_profile = profile_a if favorite_is_a else profile_b
    underdog_profile = profile_b if favorite_is_a else profile_a
    favorite_name = team_a if favorite_is_a else team_b
    underdog_name = team_b if favorite_is_a else team_a
    favorite_labels = favorite_profile.derived_labels if favorite_profile is not None else frozenset()
    underdog_labels = underdog_profile.derived_labels if underdog_profile is not None else frozenset()
    labels: set[str] = set()
    if (
        "control_team" in favorite_labels
        and underdog_labels & {"defensive_resistance_team", "low_event_team"}
    ):
        labels.add("controlled_favorite")
    if (
        "low_event_team" in favorite_labels
        and underdog_labels & {"low_event_team", "defensive_resistance_team"}
    ):
        labels.add("low_event")
    if (
        favorite_profile is not None
        and favorite_profile.strong_defense_attack_suppression_score >= 0.35
        and underdog_labels & {"low_event_team", "defensive_resistance_team"}
    ):
        labels.add("strong_defense_attack_suppression")
    if "transition_route_team" in underdog_labels:
        labels.add("transition_dog")
    if (
        ("open_event_team" in favorite_labels and underdog_labels & {"open_event_team", "defensive_fragility_team"})
        or ("open_event_team" in underdog_labels and "defensive_fragility_team" in favorite_labels)
    ):
        labels.add("open_game")
    if max(p_a, p_b) >= 0.62 and "defensive_fragility_team" in underdog_labels:
        labels.add("collapse_risk")
    if not labels:
        return None

    draw_multiplier = 1.0
    tempo_multiplier = 1.0
    favorite_attack_multiplier = 1.0
    underdog_attack_multiplier = 1.0
    if profile_a is not None:
        draw_multiplier *= profile_a.draw_multiplier
        tempo_multiplier *= profile_a.tempo_multiplier
    if profile_b is not None:
        draw_multiplier *= profile_b.draw_multiplier
        tempo_multiplier *= profile_b.tempo_multiplier
    if favorite_profile is not None:
        favorite_attack_multiplier *= favorite_profile.attack_multiplier
        underdog_attack_multiplier *= favorite_profile.opponent_attack_multiplier
        if (
            favorite_profile.strong_defense_attack_suppression_score >= 0.35
            and underdog_labels & {"low_event_team", "defensive_resistance_team"}
        ):
            favorite_attack_multiplier *= clamp_value(
                1.0 - favorite_profile.strong_defense_attack_suppression_score * 0.04,
                0.92,
                1.0,
            )
    if underdog_profile is not None:
        underdog_attack_multiplier *= underdog_profile.attack_multiplier
        favorite_attack_multiplier *= underdog_profile.opponent_attack_multiplier

    notes = (
        f"球队形态画像推导：热门={favorite_name}，弱势方={underdog_name}；"
        f"{team_a}={','.join(sorted(profile_a.derived_labels)) if profile_a else '无'}；"
        f"{team_b}={','.join(sorted(profile_b.derived_labels)) if profile_b else '无'}"
    )
    return MatchShapeContext(
        pre_match_shapes=joined_shape_labels(labels),
        observed_shapes="",
        draw_multiplier=clamp_value(draw_multiplier, 0.96, 1.06),
        tempo_multiplier=clamp_value(tempo_multiplier, 0.95, 1.06),
        favorite_attack_multiplier=clamp_value(favorite_attack_multiplier, 0.94, 1.06),
        underdog_attack_multiplier=clamp_value(underdog_attack_multiplier, 0.94, 1.08),
        notes=notes,
    )


def load_predictions() -> list[dict]:
    with PREDICTIONS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    include_completed = {
        item.strip()
        for item in os.environ.get("WC_INCLUDE_COMPLETED_MATCH_KEYS", "").split(",")
        if item.strip()
    }
    if include_completed:
        return [
            row
            for row in rows
            if f"{row['date_bjt']} {row['time_bjt']} {row['team_a']} vs {row['team_b']}" in include_completed
        ]
    return rows


def parse_bjt_datetime(date_text: str, time_text: str) -> datetime:
    return datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H:%M")


def match_group_lookup() -> dict[tuple[str, str, str, str], str]:
    lookup: dict[tuple[str, str, str, str], str] = {}
    for match in schedule():
        kickoff = match_datetime_bjt(match)
        lookup[
            (
                kickoff.strftime("%Y-%m-%d"),
                kickoff.strftime("%H:%M"),
                match.team_a,
                match.team_b,
            )
        ] = match.group
    return lookup


def load_completed_matches() -> list[CompletedMatch]:
    if not RESULTS_CSV.exists():
        return []
    groups = match_group_lookup()
    matches: list[CompletedMatch] = []
    with RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["date_bjt"], row["time_bjt"], row["team_a"], row["team_b"])
            group = row.get("group") or groups.get(key)
            if group is None:
                raise RuntimeError(f"cannot find group for completed match: {key}")
            matches.append(
                CompletedMatch(
                    kickoff_bjt=parse_bjt_datetime(row["date_bjt"], row["time_bjt"]),
                    group=group,
                    team_a=row["team_a"],
                    team_b=row["team_b"],
                    goals_a=int(row["goals_a"]),
                    goals_b=int(row["goals_b"]),
                )
            )
    return matches


def scored_match_from_completed(match: CompletedMatch) -> ScoredMatch:
    return ScoredMatch(
        kickoff_bjt=match.kickoff_bjt,
        team_a=canonical_team(match.team_a),
        team_b=canonical_team(match.team_b),
        goals_a=match.goals_a,
        goals_b=match.goals_b,
    )


@lru_cache(maxsize=1)
def load_historical_scored_matches() -> tuple[ScoredMatch, ...]:
    if not INTERNATIONAL_RESULTS_CSV.exists():
        return ()
    matches: list[ScoredMatch] = []
    with INTERNATIONAL_RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            home_score = row.get("home_score", "")
            away_score = row.get("away_score", "")
            if not home_score.isdigit() or not away_score.isdigit():
                continue
            kickoff = datetime.strptime(row["date"], "%Y-%m-%d")
            matches.append(
                ScoredMatch(
                    kickoff_bjt=kickoff,
                    team_a=canonical_team(row["home_team"]),
                    team_b=canonical_team(row["away_team"]),
                    goals_a=int(home_score),
                    goals_b=int(away_score),
                )
            )
    return tuple(matches)


def recent_goal_signal(
    team: str,
    kickoff_bjt: datetime,
    completed_matches: list[CompletedMatch],
) -> RecentGoalSignal:
    canonical = canonical_team(team)
    candidates = [
        match
        for match in load_historical_scored_matches()
        if match.kickoff_bjt < kickoff_bjt and canonical in {match.team_a, match.team_b}
    ]
    candidates.extend(
        match
        for match in (scored_match_from_completed(result) for result in completed_matches)
        if match.kickoff_bjt < kickoff_bjt and canonical in {match.team_a, match.team_b}
    )
    deduped = {
        (match.kickoff_bjt, match.team_a, match.team_b, match.goals_a, match.goals_b): match
        for match in candidates
    }
    latest = sorted(deduped.values(), key=lambda match: match.kickoff_bjt, reverse=True)[
        :RECENT_GOAL_LOOKBACK_MATCHES
    ]
    totals = tuple(match.goals_a + match.goals_b for match in latest)
    return RecentGoalSignal(team=canonical, totals=totals)


def match_favorite_team(row: dict, p_a: float, p_b: float) -> str:
    return row["team_a"] if p_a >= p_b else row["team_b"]


def empty_standings_for_group(group: str) -> dict[str, GroupStanding]:
    teams = sorted({team for match in schedule() if match.group == group for team in (match.team_a, match.team_b)})
    return {team: GroupStanding() for team in teams}


def add_result_to_standings(
    standings: dict[str, GroupStanding],
    team_a: str,
    team_b: str,
    goals_a: int,
    goals_b: int,
) -> None:
    points_a = 3 if goals_a > goals_b else 1 if goals_a == goals_b else 0
    points_b = 3 if goals_b > goals_a else 1 if goals_a == goals_b else 0
    standing_a = standings[team_a]
    standing_b = standings[team_b]
    standings[team_a] = GroupStanding(
        played=standing_a.played + 1,
        points=standing_a.points + points_a,
        goals_for=standing_a.goals_for + goals_a,
        goals_against=standing_a.goals_against + goals_b,
    )
    standings[team_b] = GroupStanding(
        played=standing_b.played + 1,
        points=standing_b.points + points_b,
        goals_for=standing_b.goals_for + goals_b,
        goals_against=standing_b.goals_against + goals_a,
    )


def standings_before_match(row: dict, completed_matches: list[CompletedMatch]) -> dict[str, GroupStanding]:
    kickoff = parse_bjt_datetime(row["date_bjt"], row["time_bjt"])
    group = row["group"]
    standings = empty_standings_for_group(group)
    for result in completed_matches:
        if result.group != group or result.kickoff_bjt >= kickoff:
            continue
        add_result_to_standings(standings, result.team_a, result.team_b, result.goals_a, result.goals_b)
    return standings


def completed_match_key(result: CompletedMatch) -> tuple[str, str, str, str]:
    return (
        result.kickoff_bjt.strftime("%Y-%m-%d"),
        result.kickoff_bjt.strftime("%H:%M"),
        result.team_a,
        result.team_b,
    )


def missing_prior_group_matches(row: dict, completed_matches: list[CompletedMatch]) -> list[str]:
    kickoff = parse_bjt_datetime(row["date_bjt"], row["time_bjt"])
    completed_keys = {completed_match_key(result) for result in completed_matches}
    missing: list[str] = []
    for match in schedule():
        match_kickoff = match_datetime_bjt(match)
        if match.group != row["group"] or match_kickoff >= kickoff:
            continue
        key = (
            match_kickoff.strftime("%Y-%m-%d"),
            match_kickoff.strftime("%H:%M"),
            match.team_a,
            match.team_b,
        )
        if key not in completed_keys:
            missing.append(f"{match.team_a} vs {match.team_b}")
    return missing


def team_draw_acceptance(standing: GroupStanding, round_number: int) -> float:
    if round_number <= 1:
        return 1.12
    if round_number == 2:
        if standing.points >= 3:
            return 1.16
        if standing.points == 1:
            return 1.02
        return 0.82
    if standing.points >= 6:
        return 1.20
    if standing.points == 4:
        return 1.18
    if standing.points == 3:
        return 0.98 if standing.goal_diff >= 0 else 0.90
    if standing.points == 2:
        return 0.86
    if standing.points == 1:
        return 0.78
    return 0.70


def apply_knockout_route_pressure(
    group: str,
    acceptance_a: float,
    acceptance_b: float,
    standing_a: GroupStanding,
    standing_b: GroupStanding,
    round_number: int,
) -> tuple[float, float, str]:
    if round_number < 3:
        return acceptance_a, acceptance_b, ""
    route = GROUP_KNOCKOUT_ROUTE_PRESSURE.get(group)
    if not route:
        return acceptance_a, acceptance_b, ""
    if standing_a.points < 6 or standing_b.points < 6:
        return acceptance_a, acceptance_b, ""
    if standing_a.points != standing_b.points:
        return acceptance_a, acceptance_b, ""

    pressure = float(route["pressure"])
    opponent = str(route["runner_up_opponent"])
    if standing_a.goal_diff > standing_b.goal_diff:
        return min(acceptance_a, 1.14), max(0.82, acceptance_b - pressure), f"第二名路线可能遇到{opponent}，落后方争头名"
    if standing_b.goal_diff > standing_a.goal_diff:
        return max(0.82, acceptance_a - pressure), min(acceptance_b, 1.14), f"第二名路线可能遇到{opponent}，落后方争头名"
    adjusted = max(0.90, min(acceptance_a, acceptance_b) - pressure * 0.5)
    return adjusted, adjusted, f"第二名路线可能遇到{opponent}，双方争头名"


def group_stage_context(row: dict, completed_matches: list[CompletedMatch]) -> GroupStageContext:
    if is_knockout_stage(row["group"]):
        return GroupStageContext(
            round_number=4,
            draw_acceptance_a=0.90,
            draw_acceptance_b=0.90,
            draw_multiplier=0.92,
            tempo_multiplier=0.98,
            attack_multiplier_a=1.0,
            attack_multiplier_b=1.0,
            opponent_attack_multiplier_a=1.0,
            opponent_attack_multiplier_b=1.0,
            open_tail_total_goals=0.10,
            qualified_a=False,
            qualified_b=False,
            complete=True,
            notes="淘汰赛；90分钟可打平但不能接受平局晋级，常规时间更重视不先犯错，加时/点球风险独立存在",
        )
    standings = standings_before_match(row, completed_matches)
    team_a = row["team_a"]
    team_b = row["team_b"]
    standing_a = standings[team_a]
    standing_b = standings[team_b]
    round_number = max(standing_a.played, standing_b.played) + 1
    missing_prior = missing_prior_group_matches(row, completed_matches)
    acceptance_a = team_draw_acceptance(standing_a, round_number)
    acceptance_b = team_draw_acceptance(standing_b, round_number)
    acceptance_a, acceptance_b, route_note = apply_knockout_route_pressure(
        row["group"],
        acceptance_a,
        acceptance_b,
        standing_a,
        standing_b,
        round_number,
    )
    qualified_a = round_number >= 3 and standing_a.points >= 6
    qualified_b = round_number >= 3 and standing_b.points >= 6
    average_acceptance = (acceptance_a + acceptance_b) / 2
    draw_multiplier = max(
        GROUP_STAGE_DRAW_MIN_MULTIPLIER,
        min(GROUP_STAGE_DRAW_MAX_MULTIPLIER, average_acceptance),
    )
    tempo_multiplier = max(
        GROUP_STAGE_TEMPO_MIN_MULTIPLIER,
        min(GROUP_STAGE_TEMPO_MAX_MULTIPLIER, 1.0 - (average_acceptance - 1.0) * HIGH_DRAW_ACCEPTANCE_TEMPO_WEIGHT),
    )
    top_seed_race = qualified_a and qualified_b and min(acceptance_a, acceptance_b) < 1.05
    if qualified_a and qualified_b:
        tempo_multiplier *= (
            GROUP_STAGE_QUALIFIED_TEMPO_MULTIPLIER
            if top_seed_race
            else GROUP_STAGE_BOTH_QUALIFIED_TEMPO_MULTIPLIER
        )
    elif qualified_a or qualified_b:
        must_chase = min(acceptance_a, acceptance_b) <= 0.82
        tempo_multiplier *= (
            GROUP_STAGE_QUALIFIED_TEMPO_WITH_CHASER_MULTIPLIER
            if must_chase
            else GROUP_STAGE_QUALIFIED_TEMPO_MULTIPLIER
        )
    tempo_multiplier = max(GROUP_STAGE_TEMPO_MIN_MULTIPLIER, min(GROUP_STAGE_TEMPO_MAX_MULTIPLIER, tempo_multiplier))
    attack_multiplier_a = GROUP_STAGE_QUALIFIED_ATTACK_MULTIPLIER if qualified_a and acceptance_a >= 1.05 else 1.0
    attack_multiplier_b = GROUP_STAGE_QUALIFIED_ATTACK_MULTIPLIER if qualified_b and acceptance_b >= 1.05 else 1.0
    opponent_attack_multiplier_a = 1.0
    opponent_attack_multiplier_b = 1.0
    open_tail_total_goals = 0.0
    if qualified_a != qualified_b:
        open_tail_total_goals = GROUP_STAGE_QUALIFIED_OPEN_TAIL_TOTAL_GOALS
        if qualified_a:
            opponent_attack_multiplier_b *= GROUP_STAGE_QUALIFIED_DEFENSE_LEAK_MULTIPLIER
            if acceptance_b <= 0.82:
                attack_multiplier_b *= GROUP_STAGE_QUALIFIED_CHASER_ATTACK_MULTIPLIER
        else:
            opponent_attack_multiplier_a *= GROUP_STAGE_QUALIFIED_DEFENSE_LEAK_MULTIPLIER
            if acceptance_a <= 0.82:
                attack_multiplier_a *= GROUP_STAGE_QUALIFIED_CHASER_ATTACK_MULTIPLIER
    notes = (
        f"第{round_number}轮；"
        f"{team_a}赛前{standing_a.points}分{standing_a.goal_diff:+d}净胜球，平局接受度{acceptance_a:.2f}；"
        f"{team_b}赛前{standing_b.points}分{standing_b.goal_diff:+d}净胜球，平局接受度{acceptance_b:.2f}"
    )
    if route_note:
        notes += f"；{route_note}"
    qualified_notes = []
    if qualified_a:
        qualified_notes.append(f"{team_a}已提前出线，进攻按轮换/保体能下调，防守稳定性下调")
    if qualified_b:
        qualified_notes.append(f"{team_b}已提前出线，进攻按轮换/保体能下调，防守稳定性下调")
    if qualified_notes:
        notes += "；" + "；".join(qualified_notes)
    if missing_prior:
        notes += f"；同组前序赛果缺{len(missing_prior)}场，未使用未知赛果，只按已录入赛果计算"
    return GroupStageContext(
        round_number=round_number,
        draw_acceptance_a=acceptance_a,
        draw_acceptance_b=acceptance_b,
        draw_multiplier=draw_multiplier,
        tempo_multiplier=tempo_multiplier,
        attack_multiplier_a=attack_multiplier_a,
        attack_multiplier_b=attack_multiplier_b,
        opponent_attack_multiplier_a=opponent_attack_multiplier_a,
        opponent_attack_multiplier_b=opponent_attack_multiplier_b,
        open_tail_total_goals=open_tail_total_goals,
        qualified_a=qualified_a,
        qualified_b=qualified_b,
        complete=not missing_prior,
        notes=notes,
    )


def parse_score(value: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)-(\d+)$", value)
    if not match:
        raise ValueError(f"invalid score: {value}")
    return int(match.group(1)), int(match.group(2))


def format_score(cell: tuple[int, int, float]) -> str:
    return f"{cell[0]}-{cell[1]}"


def score_matches_outcome(goals_a: int, goals_b: int, outcome: str) -> bool:
    if outcome == "A":
        return goals_a > goals_b
    if outcome == "B":
        return goals_a < goals_b
    if outcome == "D":
        return goals_a == goals_b
    raise ValueError(f"unknown outcome: {outcome}")


def best_score(
    cells: list[tuple[int, int, float]],
    bucket_labels: set[str],
    outcomes: set[str] | None = None,
) -> tuple[int, int, float]:
    for goals_a, goals_b, probability in cells:
        if total_goal_bucket(goals_a + goals_b) not in bucket_labels:
            continue
        if outcomes is None:
            return goals_a, goals_b, probability
        score_outcome = "A" if goals_a > goals_b else "B" if goals_a < goals_b else "D"
        if score_outcome in outcomes:
            return goals_a, goals_b, probability
    raise RuntimeError(f"no score found for buckets={bucket_labels}, outcomes={outcomes}")


def best_distinct_score(
    cells: list[tuple[int, int, float]],
    bucket_labels: set[str],
    outcomes: set[str] | None,
    excluded_scores: set[tuple[int, int]],
) -> tuple[int, int, float]:
    for goals_a, goals_b, probability in cells:
        if (goals_a, goals_b) in excluded_scores:
            continue
        if total_goal_bucket(goals_a + goals_b) not in bucket_labels:
            continue
        if outcomes is None:
            return goals_a, goals_b, probability
        score_outcome = "A" if goals_a > goals_b else "B" if goals_a < goals_b else "D"
        if score_outcome in outcomes:
            return goals_a, goals_b, probability
    raise RuntimeError(
        f"no distinct score found for buckets={bucket_labels}, outcomes={outcomes}, excluded={excluded_scores}"
    )


def context_is_high_risk(
    context_a: TeamContext,
    context_b: TeamContext,
    p_a: float,
    p_draw: float,
    p_b: float,
) -> bool:
    top = max(p_a, p_draw, p_b)
    second = sorted([p_a, p_draw, p_b], reverse=True)[1]
    max_temperature = max(
        float(context_a.weather_high_c or 0),
        float(context_b.weather_high_c or 0),
    )
    has_heat_penalty = max_temperature >= 28.0 and (
        context_a.weather_multiplier < 0.98 or context_b.weather_multiplier < 0.98
    )
    has_strong_heat_penalty = max_temperature >= 28.0 and (
        context_a.weather_multiplier <= 0.95 or context_b.weather_multiplier <= 0.95
    )
    has_travel_penalty = context_a.travel_multiplier < 0.98 or context_b.travel_multiplier < 0.98
    has_injury_penalty = context_a.injury_multiplier < 0.98 or context_b.injury_multiplier < 0.98
    return (
        (top - second < 0.10)
        or has_strong_heat_penalty
        or (p_draw >= 0.25 and (has_heat_penalty or has_travel_penalty or has_injury_penalty))
    )


def context_tempo_discount(
    context_a: TeamContext,
    context_b: TeamContext,
    p_a: float,
    p_draw: float,
    p_b: float,
) -> float:
    max_temperature = max(
        float(context_a.weather_high_c or 0),
        float(context_b.weather_high_c or 0),
    )
    strong_favorite = max(p_a, p_b) >= STRONG_FAVORITE_PROBABILITY
    if max_temperature >= 28.0 and (
        context_a.weather_multiplier <= 0.95 or context_b.weather_multiplier <= 0.95
    ):
        return STRONG_CONTEXT_TEMPO_DISCOUNT
    if not strong_favorite and context_is_high_risk(context_a, context_b, p_a, p_draw, p_b):
        return HIGH_CONTEXT_TEMPO_DISCOUNT
    return 1.0


def normalize_probabilities(p_a: float, p_draw: float, p_b: float) -> tuple[float, float, float]:
    total = p_a + p_draw + p_b
    if total <= 0:
        raise RuntimeError("outcome probabilities sum to zero")
    return p_a / total, p_draw / total, p_b / total


def shape_label_set(shape: MatchShapeContext | None) -> set[str]:
    if shape is None:
        return set()
    return {label for label in shape.pre_match_shapes.split(";") if label}


def style_edge_side(labels: set[str]) -> str | None:
    has_a = STYLE_EDGE_LABEL_A in labels
    has_b = STYLE_EDGE_LABEL_B in labels
    if has_a and has_b:
        raise RuntimeError("style edge cannot point to both teams")
    if has_a:
        return "A"
    if has_b:
        return "B"
    return None


def realtime_style_features(row: dict, side: str, profile: TeamShapeProfile | None) -> frozenset[str]:
    features = set(prediction_row_style_features(row, side))
    if profile is not None:
        features.update(team_shape_style_features(profile.derived_labels))
    return frozenset(sorted(features))


def apply_style_matchup_probabilities(
    p_a: float,
    p_draw: float,
    p_b: float,
    effect_edge: float,
    scale_a: float,
    scale_b: float,
) -> tuple[float, float, float]:
    if abs(effect_edge) < 0.001:
        return p_a, p_draw, p_b
    p_a *= scale_a
    p_b *= scale_b
    p_draw *= clamp_value(1.0 - abs(effect_edge) * 0.75, 0.92, 1.00)
    return normalize_probabilities(p_a, p_draw, p_b)


def apply_style_matchup_xg(
    lambda_a: float,
    lambda_b: float,
    scale_a: float,
    scale_b: float,
    total_multiplier: float,
) -> tuple[float, float]:
    return (
        lambda_a * scale_a * total_multiplier,
        lambda_b * scale_b * total_multiplier,
    )


def shape_adjusted_probabilities(
    p_a: float,
    p_draw: float,
    p_b: float,
    shape: MatchShapeContext | None,
) -> tuple[float, float, float]:
    if shape is None:
        return p_a, p_draw, p_b
    labels = shape_label_set(shape)
    p_draw = max(MIN_DRAW_PROBABILITY, min(MAX_DRAW_PROBABILITY, p_draw * shape.draw_multiplier))
    if p_a >= p_b:
        p_a *= shape.favorite_attack_multiplier
        p_b *= shape.underdog_attack_multiplier
    else:
        p_a *= shape.underdog_attack_multiplier
        p_b *= shape.favorite_attack_multiplier
    side = style_edge_side(labels)
    if side == "A":
        p_a *= STYLE_EDGE_FAVORED_PROBABILITY_MULTIPLIER
        p_b *= STYLE_EDGE_SUPPRESSED_PROBABILITY_MULTIPLIER
        p_draw *= STYLE_EDGE_DRAW_PROBABILITY_MULTIPLIER
    elif side == "B":
        p_a *= STYLE_EDGE_SUPPRESSED_PROBABILITY_MULTIPLIER
        p_b *= STYLE_EDGE_FAVORED_PROBABILITY_MULTIPLIER
        p_draw *= STYLE_EDGE_DRAW_PROBABILITY_MULTIPLIER
    p_draw = max(MIN_DRAW_PROBABILITY, min(MAX_DRAW_PROBABILITY, p_draw))
    non_draw = max(0.0, 1.0 - p_draw)
    old_non_draw = p_a + p_b
    if old_non_draw <= 0:
        return normalize_probabilities(non_draw / 2, p_draw, non_draw / 2)
    return normalize_probabilities(non_draw * p_a / old_non_draw, p_draw, non_draw * p_b / old_non_draw)


def key_player_effect(
    team: str,
    team_rank: int,
    match_key: str,
    match_name: str,
    context: TeamContext | None,
    signals: dict[str, KeyPlayerSignal],
    statuses: dict[tuple[str, str, str], KeyPlayerMatchStatus],
    team_market_values: dict,
) -> KeyPlayerEffect:
    canonical = canonical_team(team)
    signal = signals.get(canonical)
    if signal is None:
        return KeyPlayerEffect(0.0, "")
    status = statuses.get((match_key, match_name, canonical))
    if status is not None:
        impact = clamp_value(status.impact, -1.0, 1.0)
        team_value = team_market_values.get(canonical)
        if signal.override_value_share is not None:
            value_share = signal.override_value_share
            value_scale = clamp_value(
                value_share / STAR_VALUE_SHARE_REFERENCE,
                STAR_VALUE_SHARE_MIN_SCALE,
                STAR_VALUE_SHARE_MAX_SCALE,
            )
        elif team_value is None or team_value.total_eur_m <= 0:
            value_share = 0.0
            value_scale = 1.0
        else:
            value_share = signal.market_value_eur_m / team_value.total_eur_m
            value_scale = clamp_value(
                value_share / STAR_VALUE_SHARE_REFERENCE,
                STAR_VALUE_SHARE_MIN_SCALE,
                STAR_VALUE_SHARE_MAX_SCALE,
            )
        rank_scale = star_rank_goal_scale(team_rank)
        goal_bonus = clamp_value(
            signal.goal_bonus * value_scale * rank_scale * impact,
            STAR_GOAL_BONUS_MIN,
            STAR_GOAL_BONUS_MAX,
        )
        label = (
            f"{signal.key_player}:{status.status}:{impact:.2f}:"
            f"share{value_share:.2f}:scale{value_scale:.2f}:rank{team_rank}:goal{goal_bonus:+.2f}"
        )
        return KeyPlayerEffect(
            goal_bonus,
            label,
        )

    return KeyPlayerEffect(0.0, "")


def star_rank_goal_scale(team_rank: int) -> float:
    return clamp_value(
        STAR_RANK_GOAL_SCALE_MIN + max(0, team_rank - 1) / STAR_RANK_GOAL_SCALE_SPAN,
        STAR_RANK_GOAL_SCALE_MIN,
        STAR_RANK_GOAL_SCALE_MAX,
    )


def apply_key_player_team_goals(
    lambda_a: float,
    lambda_b: float,
    effect_a: KeyPlayerEffect,
    effect_b: KeyPlayerEffect,
) -> tuple[float, float]:
    return lambda_a * (1.0 + effect_a.goal_bonus), lambda_b * (1.0 + effect_b.goal_bonus)


def apply_shape_attack(
    lambda_a: float,
    lambda_b: float,
    p_a: float,
    p_b: float,
    shape: MatchShapeContext | None,
) -> tuple[float, float]:
    if shape is None:
        return lambda_a, lambda_b
    labels = shape_label_set(shape)
    if p_a >= p_b:
        lambda_a, lambda_b = (
            lambda_a * shape.favorite_attack_multiplier,
            lambda_b * shape.underdog_attack_multiplier,
        )
    else:
        lambda_a, lambda_b = (
            lambda_a * shape.underdog_attack_multiplier,
            lambda_b * shape.favorite_attack_multiplier,
        )
    side = style_edge_side(labels)
    if side == "A":
        return lambda_a * STYLE_EDGE_FAVORED_XG_MULTIPLIER, lambda_b * STYLE_EDGE_SUPPRESSED_XG_MULTIPLIER
    if side == "B":
        return lambda_a * STYLE_EDGE_SUPPRESSED_XG_MULTIPLIER, lambda_b * STYLE_EDGE_FAVORED_XG_MULTIPLIER
    return lambda_a, lambda_b


def apply_group_stage_probabilities(
    p_a: float,
    p_draw: float,
    p_b: float,
    group_context: GroupStageContext,
) -> tuple[float, float, float]:
    p_draw = max(MIN_DRAW_PROBABILITY, min(MAX_DRAW_PROBABILITY, p_draw * group_context.draw_multiplier))
    non_draw = max(0.0, 1.0 - p_draw)
    old_non_draw = p_a + p_b
    if old_non_draw <= 0:
        return normalize_probabilities(non_draw / 2, p_draw, non_draw / 2)
    return normalize_probabilities(non_draw * p_a / old_non_draw, p_draw, non_draw * p_b / old_non_draw)


def predicted_outcome_with_draw_acceptance(
    p_a: float,
    p_draw: float,
    p_b: float,
    group_context: GroupStageContext,
) -> str:
    probabilities = sorted([p_a, p_draw, p_b], reverse=True)
    if probabilities[0] - probabilities[2] <= THREE_WAY_CLOSE_MARGIN:
        return "D"
    top_non_draw = max(p_a, p_b)
    if group_context.draw_multiplier > 1.0 and top_non_draw - p_draw <= DRAW_CLOSE_TO_TOP_MARGIN:
        return "D"
    if p_a >= p_draw and p_a >= p_b:
        return "A"
    if p_b >= p_draw:
        return "B"
    return "D"


def raw_xg_outcome_probabilities(lambda_a: float, lambda_b: float) -> tuple[float, float, float]:
    p_a = p_draw = p_b = 0.0
    total = 0.0
    for goals_a in range(MAX_GOALS + 1):
        for goals_b in range(MAX_GOALS + 1):
            probability = poisson_pmf(goals_a, lambda_a) * poisson_pmf(goals_b, lambda_b)
            total += probability
            if goals_a > goals_b:
                p_a += probability
            elif goals_b > goals_a:
                p_b += probability
            else:
                p_draw += probability
    if total <= 0:
        raise RuntimeError("raw xG outcome probability sum is zero")
    return normalize_probabilities(p_a / total, p_draw / total, p_b / total)


def xg_outcome_probability_edge(lambda_a: float, lambda_b: float, predicted_outcome: str) -> float:
    p_a, p_draw, p_b = raw_xg_outcome_probabilities(lambda_a, lambda_b)
    probabilities = {"A": p_a, "D": p_draw, "B": p_b}
    selected = probabilities[predicted_outcome]
    return selected - max(value for key, value in probabilities.items() if key != predicted_outcome)


def legacy_outcome_probability_edge(p_a: float, p_draw: float, p_b: float, predicted_outcome: str) -> float:
    probabilities = {"A": p_a, "D": p_draw, "B": p_b}
    selected = probabilities[predicted_outcome]
    return selected - max(value for key, value in probabilities.items() if key != predicted_outcome)


def legacy_top_outcome(p_a: float, p_draw: float, p_b: float) -> str:
    return max([("A", p_a), ("D", p_draw), ("B", p_b)], key=lambda item: item[1])[0]


def outcome_edge_conflict(
    lambda_a: float,
    lambda_b: float,
    p_a: float,
    p_draw: float,
    p_b: float,
    predicted_outcome: str,
) -> bool:
    xg_edge = xg_outcome_probability_edge(lambda_a, lambda_b, predicted_outcome)
    legacy_edge = legacy_outcome_probability_edge(p_a, p_draw, p_b, predicted_outcome)
    return (
        legacy_top_outcome(p_a, p_draw, p_b) != predicted_outcome
        or abs(xg_edge - legacy_edge) >= XG_OUTCOME_PROBABILITY_CONFLICT
    )


def xg_draw_margin_threshold(group_context: GroupStageContext, shape_labels: str) -> float:
    factor = 1.0
    if (
        group_context.round_number == 1
        and group_context.draw_acceptance_a >= 1.05
        and group_context.draw_acceptance_b >= 1.05
    ):
        factor *= XG_DRAW_FIRST_ROUND_BOTH_NEUTRAL_FACTOR
    elif group_context.draw_acceptance_a >= 1.05 and group_context.draw_acceptance_b >= 1.05:
        factor *= XG_DRAW_ACCEPTABLE_FACTOR
    elif min(group_context.draw_acceptance_a, group_context.draw_acceptance_b) < 0.95:
        factor *= XG_DRAW_MUST_CHASE_FACTOR

    labels = {label for label in shape_labels.split(";") if label}
    if labels & {"low_block", "low_event", "low_event_favorite"}:
        factor *= XG_DRAW_LOW_EVENT_FACTOR
    if labels & {"open_game", "open_mismatch", "collapse_risk"}:
        factor *= XG_DRAW_OPEN_GAME_FACTOR
    if style_edge_side(labels) is not None:
        factor *= STYLE_EDGE_DRAW_MARGIN_FACTOR

    if group_context.tempo_multiplier < 0.96:
        factor *= XG_DRAW_LOW_TEMPO_FACTOR
    elif group_context.tempo_multiplier > 1.04:
        factor *= XG_DRAW_HIGH_TEMPO_FACTOR

    return XG_DRAW_BASE_MARGIN * factor


def predicted_outcome_from_xg(
    lambda_a: float,
    lambda_b: float,
    group_context: GroupStageContext,
    shape_labels: str,
) -> str:
    p_a, p_draw, p_b = raw_xg_outcome_probabilities(lambda_a, lambda_b)
    top_non_draw = max(p_a, p_b)
    if top_non_draw - p_draw <= xg_draw_margin_threshold(group_context, shape_labels):
        return "D"
    return "A" if p_a > p_b else "B"


def apply_qualified_low_bucket_outcome_adjustment(
    p_a: float,
    p_draw: float,
    p_b: float,
    predicted_outcome: str,
    selected_bucket: str,
    group_context: GroupStageContext,
) -> tuple[float, float, float, str]:
    if selected_bucket != "0-1球" or group_context.round_number < 3:
        return p_a, p_draw, p_b, predicted_outcome

    top_outcome = "A" if p_a >= p_b else "B"
    qualified_top = (top_outcome == "A" and group_context.qualified_a) or (
        top_outcome == "B" and group_context.qualified_b
    )
    if not qualified_top:
        return p_a, p_draw, p_b, predicted_outcome

    top_non_draw = max(p_a, p_b)
    if (
        p_draw < QUALIFIED_LOW_BUCKET_DRAW_MIN_PROBABILITY
        or top_non_draw - p_draw > QUALIFIED_LOW_BUCKET_DRAW_MAX_EDGE
    ):
        return p_a, p_draw, p_b, predicted_outcome

    if top_outcome == "A":
        p_a *= QUALIFIED_LOW_BUCKET_WIN_DISCOUNT
    else:
        p_b *= QUALIFIED_LOW_BUCKET_WIN_DISCOUNT
    p_draw *= QUALIFIED_LOW_BUCKET_DRAW_BOOST
    p_a, p_draw, p_b = normalize_probabilities(p_a, p_draw, p_b)
    return p_a, p_draw, p_b, "D"


def apply_knockout_draw_score_adjustment(
    p_a: float,
    p_draw: float,
    p_b: float,
    predicted_outcome: str,
    selected_bucket: str,
    group_context: GroupStageContext,
    shape_labels: str,
    lambda_a: float,
    lambda_b: float,
) -> str:
    if group_context.round_number != 4 or selected_bucket == "0-1球":
        return predicted_outcome

    top_non_draw = max(p_a, p_b)
    edge = top_non_draw - p_draw
    xg_p_a, xg_p_draw, xg_p_b = raw_xg_outcome_probabilities(lambda_a, lambda_b)
    xg_top = "A" if xg_p_a >= xg_p_b else "B"
    xg_draw_veto = (
        predicted_outcome != "D"
        and xg_top == predicted_outcome
        and max(xg_p_a, xg_p_b) - xg_p_draw > KNOCKOUT_DRAW_XG_VETO_MARGIN
    )
    if edge <= KNOCKOUT_DRAW_CLOSE_MARGIN and not xg_draw_veto:
        return "D"

    labels = {label for label in shape_labels.split(";") if label}
    if (
        labels & KNOCKOUT_LOW_EVENT_DRAW_LABELS
        and p_draw >= KNOCKOUT_LOW_EVENT_MIN_DRAW_PROBABILITY
        and edge <= KNOCKOUT_LOW_EVENT_DRAW_MARGIN
        and not xg_draw_veto
    ):
        return "D"
    return predicted_outcome


def apply_style_matchup_outcome_adjustment(
    predicted_outcome: str,
    style_edge: float,
    p_a: float,
    p_draw: float,
    p_b: float,
    lambda_a: float,
    lambda_b: float,
    shape_labels: str,
) -> str:
    if predicted_outcome != "D":
        return predicted_outcome
    if abs(style_edge) < 0.060:
        return predicted_outcome
    labels = {label for label in shape_labels.split(";") if label}
    if labels & {"controlled_favorite", "low_event_favorite"}:
        return predicted_outcome
    target = "A" if style_edge > 0 else "B"
    target_probability = p_a if target == "A" else p_b
    other_probability = p_b if target == "A" else p_a
    target_xg = lambda_a if target == "A" else lambda_b
    other_xg = lambda_b if target == "A" else lambda_a
    if target_probability + 0.01 < other_probability:
        return predicted_outcome
    if target_probability + 0.04 < p_draw:
        return predicted_outcome
    if target_xg + 0.05 < other_xg:
        return predicted_outcome
    return target


def qualified_upset_risk(group_context: GroupStageContext) -> bool:
    if group_context.round_number < 3 or group_context.qualified_a == group_context.qualified_b:
        return False
    if group_context.qualified_a:
        return group_context.draw_acceptance_b <= QUALIFIED_UPSET_RISK_MAX_CHASER_DRAW_ACCEPTANCE
    return group_context.draw_acceptance_a <= QUALIFIED_UPSET_RISK_MAX_CHASER_DRAW_ACCEPTANCE


def adjusted_risk_fields(row: dict, group_context: GroupStageContext) -> tuple[str, str]:
    risk_label = row.get("risk_label", "")
    risk_reasons = row.get("risk_reasons", "")
    if not qualified_upset_risk(group_context):
        return risk_label, risk_reasons

    risk_label = "可能爆冷"
    reason = "第三轮已出线强队可轮换/保体能，对手仍需抢分，属于可赢可输局"
    if risk_reasons:
        risk_reasons = f"{risk_reasons}; {reason}"
    else:
        risk_reasons = reason
    return risk_label, risk_reasons


def allow_realtime_bucket_suppression(
    original_bucket: str,
    selected_bucket: str,
    shape_labels: str,
    group_context: GroupStageContext | None = None,
) -> bool:
    if (
        selected_bucket == "0-1球"
        and group_context is not None
        and group_context.round_number >= 3
        and (group_context.qualified_a or group_context.qualified_b)
        and not (
            {label for label in shape_labels.split(";") if label}
            & {"open_game", "open_mismatch", "collapse_risk"}
        )
    ):
        return True
    if selected_bucket != "2-3球" or original_bucket not in HIGH_TOTAL_GOAL_BUCKETS:
        return True
    labels = {label for label in shape_labels.split(";") if label}
    top_seed_race = (
        group_context is not None
        and group_context.round_number >= 3
        and group_context.qualified_a
        and group_context.qualified_b
        and min(group_context.draw_acceptance_a, group_context.draw_acceptance_b) < 1.05
    )
    if (
        group_context is not None
        and group_context.round_number >= 2
        and min(group_context.draw_acceptance_a, group_context.draw_acceptance_b)
        >= GROUP_DRAW_SLOWDOWN_MIN_ACCEPTANCE_START
        and group_context.tempo_multiplier <= 0.96
        and not top_seed_race
        and not (labels & {"open_game", "open_mismatch", "collapse_risk"})
    ):
        return True
    return bool(labels & {"low_block", "low_event_favorite", "low_event", "controlled_favorite"})


def constrained_realtime_total_goal_bucket(
    original_bucket: str,
    selected_bucket: str,
    shape_labels: str,
    group_context: GroupStageContext | None = None,
) -> str:
    if allow_realtime_bucket_suppression(original_bucket, selected_bucket, shape_labels, group_context):
        return selected_bucket
    return original_bucket


def protect_strong_favorite_from_low_bucket(
    selected_bucket: str,
    total_buckets: list[tuple[str, float]],
    lambda_a: float,
    lambda_b: float,
    p_a: float,
    p_b: float,
) -> tuple[str, list[tuple[str, float]]]:
    if selected_bucket != "0-1球":
        return selected_bucket, total_buckets

    favorite_xg = max(lambda_a, lambda_b)
    underdog_xg = min(lambda_a, lambda_b)
    favorite_probability = max(p_a, p_b)
    if (
        favorite_xg < STRONG_FAVORITE_LOW_BUCKET_MIN_FAVORITE_XG
        or underdog_xg > STRONG_FAVORITE_LOW_BUCKET_MAX_UNDERDOG_XG
        or favorite_probability < STRONG_FAVORITE_LOW_BUCKET_MIN_PROBABILITY
    ):
        return selected_bucket, total_buckets

    promoted_bucket = "2-3球"
    bucket_map = {bucket: probability for bucket, probability in total_buckets}
    bucket_map[promoted_bucket] = max(bucket_map.get(promoted_bucket, 0.0), bucket_map.get(selected_bucket, 0.0) + 0.001)
    total_probability = sum(bucket_map.values())
    if total_probability <= 0:
        raise RuntimeError("total-goal bucket probabilities sum to zero after strong favorite protection")
    adjusted_buckets = sorted(
        ((bucket, probability / total_probability) for bucket, probability in bucket_map.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return promoted_bucket, adjusted_buckets


def apply_knockout_high_bucket_cap(
    stage: str,
    selected_bucket: str,
    total_buckets: list[tuple[str, float]],
    predicted_outcome: str,
    lambda_a: float,
    lambda_b: float,
    p_a: float,
    p_draw: float,
    p_b: float,
    shape_labels: str,
    group_context: GroupStageContext,
) -> tuple[str, list[tuple[str, float]], bool]:
    if group_context.round_number != 4 or selected_bucket not in {"4-5球", "6-8球"}:
        return selected_bucket, total_buckets, False

    labels = {label for label in shape_labels.split(";") if label}
    stage = stage.upper()
    is_early_stage = stage in KNOCKOUT_EARLY_HIGH_BUCKET_CAP_STAGES
    is_late_stage = stage in KNOCKOUT_LATE_HIGH_BUCKET_CAP_STAGES
    if not is_early_stage and not is_late_stage:
        return selected_bucket, total_buckets, False

    favorite_probability = max(p_a, p_b)
    favorite_xg = max(lambda_a, lambda_b)
    underdog_xg = min(lambda_a, lambda_b)
    xg_gap = abs(lambda_a - lambda_b)
    if (
        is_early_stage
        and not labels & KNOCKOUT_HIGH_BUCKET_EXTREME_EXEMPT_LABELS
        and favorite_probability >= KNOCKOUT_EARLY_CONTROL_CAP_MIN_PROBABILITY
        and favorite_xg >= KNOCKOUT_EARLY_CONTROL_CAP_MIN_FAVORITE_XG
        and xg_gap >= KNOCKOUT_EARLY_CONTROL_CAP_MIN_XG_GAP
        and underdog_xg <= KNOCKOUT_EARLY_CONTROL_CAP_MAX_UNDERDOG_XG
    ):
        capped_bucket = "2-3球"
        bucket_map = {bucket: probability for bucket, probability in total_buckets}
        old_high_probability = max(bucket_map.get("4-5球", 0.0), bucket_map.get("6-8球", 0.0))
        bucket_map[capped_bucket] = max(bucket_map.get(capped_bucket, 0.0), old_high_probability + 0.001)
        bucket_map["4-5球"] = max(
            bucket_map.get("4-5球", 0.0),
            old_high_probability * KNOCKOUT_HIGH_BUCKET_BACKUP_FACTOR,
        )
        bucket_map["6-8球"] = min(
            bucket_map.get("6-8球", 0.0),
            bucket_map["4-5球"] * KNOCKOUT_HIGH_BUCKET_TAIL_FACTOR,
        )
        total_probability = sum(bucket_map.values())
        if total_probability <= 0:
            raise RuntimeError("total-goal bucket probabilities sum to zero after early knockout control cap")
        adjusted_buckets = sorted(
            ((bucket, probability / total_probability) for bucket, probability in bucket_map.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        return capped_bucket, adjusted_buckets, True

    if selected_bucket != "6-8球" or predicted_outcome == "D":
        return selected_bucket, total_buckets, False
    if is_early_stage and labels & KNOCKOUT_TRUE_OPEN_LABELS:
        return selected_bucket, total_buckets, False
    if is_late_stage and labels & KNOCKOUT_TRUE_OPEN_LABELS and not labels & KNOCKOUT_LOW_EVENT_HIGH_BUCKET_LABELS:
        return selected_bucket, total_buckets, False

    if (
        favorite_probability < KNOCKOUT_HIGH_BUCKET_CAP_MIN_PROBABILITY
        or favorite_xg < KNOCKOUT_HIGH_BUCKET_CAP_MIN_FAVORITE_XG
        or xg_gap < KNOCKOUT_HIGH_BUCKET_CAP_MIN_XG_GAP
    ):
        return selected_bucket, total_buckets, False

    capped_bucket = "4-5球"
    if labels & KNOCKOUT_LOW_EVENT_HIGH_BUCKET_LABELS and p_draw >= KNOCKOUT_HIGH_BUCKET_LOW_EVENT_DRAW_MIN_PROBABILITY:
        capped_bucket = "2-3球"

    bucket_map = {bucket: probability for bucket, probability in total_buckets}
    old_high_probability = bucket_map.get("6-8球", 0.0)
    bucket_map[capped_bucket] = max(bucket_map.get(capped_bucket, 0.0), old_high_probability + 0.001)
    backup_bucket = "2-3球" if capped_bucket == "4-5球" else "4-5球"
    bucket_map[backup_bucket] = max(
        bucket_map.get(backup_bucket, 0.0),
        old_high_probability * KNOCKOUT_HIGH_BUCKET_BACKUP_FACTOR,
    )
    bucket_map["6-8球"] = min(
        bucket_map.get("6-8球", 0.0),
        bucket_map[backup_bucket] * KNOCKOUT_HIGH_BUCKET_TAIL_FACTOR,
    )
    total_probability = sum(bucket_map.values())
    if total_probability <= 0:
        raise RuntimeError("total-goal bucket probabilities sum to zero after knockout high bucket cap")
    adjusted_buckets = sorted(
        ((bucket, probability / total_probability) for bucket, probability in bucket_map.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    return capped_bucket, adjusted_buckets, True


def linear_score(value: float, start: float, full: float) -> float:
    if value <= start:
        return 0.0
    if value >= full:
        return 1.0
    return (value - start) / (full - start)


def group_draw_total_goal_adjustment(
    expected_total_goals: float,
    p_a: float,
    p_draw: float,
    p_b: float,
    shape_labels: str,
    group_context: GroupStageContext,
) -> float:
    labels = {label for label in shape_labels.split(";") if label}
    both_qualified = group_context.round_number >= 3 and group_context.qualified_a and group_context.qualified_b
    top_seed_race = both_qualified and min(group_context.draw_acceptance_a, group_context.draw_acceptance_b) < 1.05
    if labels & {"open_game", "open_mismatch", "collapse_risk"} and not both_qualified:
        return expected_total_goals

    average_acceptance = (group_context.draw_acceptance_a + group_context.draw_acceptance_b) / 2
    minimum_acceptance = min(group_context.draw_acceptance_a, group_context.draw_acceptance_b)
    acceptance_score = (
        linear_score(
            average_acceptance,
            GROUP_DRAW_SLOWDOWN_ACCEPTANCE_START,
            GROUP_DRAW_SLOWDOWN_ACCEPTANCE_FULL,
        )
        * linear_score(
            minimum_acceptance,
            GROUP_DRAW_SLOWDOWN_MIN_ACCEPTANCE_START,
            GROUP_DRAW_SLOWDOWN_MIN_ACCEPTANCE_FULL,
        )
    )
    draw_score = linear_score(p_draw, GROUP_DRAW_SLOWDOWN_DRAW_START, GROUP_DRAW_SLOWDOWN_DRAW_FULL)
    shape_score = 0.55
    if "credible_opponent" in labels:
        shape_score += 0.35
    if labels & {"low_block", "low_event_favorite", "low_event"}:
        shape_score += 0.20
    if labels & {"transition_dog", "set_piece_risk"}:
        shape_score += 0.10
    if both_qualified and not top_seed_race:
        shape_score = max(shape_score, 0.90)
    shape_score = min(1.0, shape_score)
    round_factor = 1.0 if group_context.round_number >= 2 else GROUP_DRAW_SLOWDOWN_ROUND_ONE_FACTOR

    slowdown = acceptance_score * draw_score * shape_score * round_factor
    if slowdown <= 0:
        return expected_total_goals
    return max(0.4, expected_total_goals * (1.0 - GROUP_DRAW_SLOWDOWN_MAX_DROP * slowdown))


def market_value_micro_adjust(
    raw_score: str,
    context_a: TeamContext,
    context_b: TeamContext,
) -> str:
    market_a, market_b = parse_score(raw_score)
    total = market_a + market_b
    if total <= 1:
        return raw_score

    max_temperature = max(
        float(context_a.weather_high_c or 0),
        float(context_b.weather_high_c or 0),
    )
    contextual_slowdown = (
        (max_temperature >= 33.0 and (context_a.weather_multiplier <= 0.95 or context_b.weather_multiplier <= 0.95))
        or context_a.injury_multiplier < 0.94
        or context_b.injury_multiplier < 0.94
    )
    target_total = min(total, MARKET_VALUE_MAX_TOTAL)
    if contextual_slowdown:
        target_total = min(target_total, total - 1)
    target_total = max(0, target_total)
    if target_total >= total:
        return raw_score

    while market_a + market_b > target_total:
        if market_a == 0 and market_b == 0:
            break
        if market_a > market_b:
            market_a -= 1
        elif market_b > market_a:
            market_b -= 1
        else:
            weaker_context = "A" if context_a.attack_multiplier <= context_b.attack_multiplier else "B"
            if weaker_context == "A" and market_a > 0:
                market_a -= 1
            elif market_b > 0:
                market_b -= 1
            else:
                market_a = max(0, market_a - 1)
    return f"{market_a}-{market_b}"


def score_outcome(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "A"
    if goals_b > goals_a:
        return "B"
    return "D"


def legal_margin_for_total(total_goals: int, raw_margin: float, outcome: str) -> int:
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


def adjusted_margin_for_market_bucket(
    total_goals: int,
    signed_margin: int,
    outcome: str,
    raw_score: str,
) -> int:
    market_a, market_b = parse_score(raw_score)
    raw_total = max(1, market_a + market_b)
    p_a = market_a / raw_total
    p_b = market_b / raw_total
    lambda_a = max(0.05, float(market_a))
    lambda_b = max(0.05, float(market_b))
    return adjust_margin_for_underdog_goal(total_goals, signed_margin, outcome, p_a, p_b, lambda_a, lambda_b)


def score_from_total_and_margin(total_goals: int, signed_margin: int) -> tuple[int, int]:
    goals_a = (total_goals + signed_margin) // 2
    goals_b = total_goals - goals_a
    return goals_a, goals_b


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


def compatible_total_goal_values(bucket: str, outcome: str) -> set[int]:
    values = total_goal_values_for_bucket(bucket)
    if outcome == "D":
        return {total for total in values if total % 2 == 0}
    return {total for total in values if total > 0}


def market_score_to_bucket(
    raw_score: str,
    bucket: str,
    total_probabilities: dict[int, float],
    margin_multiplier: float = 1.0,
) -> str:
    market_a, market_b = parse_score(raw_score)
    raw_total = market_a + market_b
    if raw_total == 0:
        raw_share = 0.5
    else:
        raw_share = market_a / raw_total
    raw_margin = raw_total * (raw_share - 0.5) * 2.0 * margin_multiplier
    if abs(raw_margin) < 0.5:
        outcome = "D"
    else:
        outcome = "A" if raw_margin > 0 else "B"
    candidates = sorted(compatible_total_goal_values(bucket, outcome) or total_goal_values_for_bucket(bucket))
    target_total = min(
        candidates,
        key=lambda total: (
            abs(total - raw_total),
            -total_probabilities.get(total, 0.0),
            total,
        ),
    )
    bucket_margin = target_total * (raw_share - 0.5) * 2.0 * margin_multiplier
    signed_margin = legal_margin_for_total(target_total, bucket_margin, outcome)
    signed_margin = adjusted_margin_for_market_bucket(target_total, signed_margin, outcome, raw_score)
    goals_a, goals_b = score_from_total_and_margin(target_total, signed_margin)
    return f"{goals_a}-{goals_b}"


def market_score_constrained_to_bucket(
    score: str,
    raw_score: str,
    bucket: str,
    total_probabilities: dict[int, float],
) -> str:
    goals_a, goals_b = parse_score(score)
    if total_goal_bucket(goals_a + goals_b) == bucket:
        return score
    return market_score_to_bucket(raw_score, bucket, total_probabilities)


def market_score_constrained_to_buckets(
    score: str,
    raw_score: str,
    bucket_options: set[str],
    total_probabilities: dict[int, float],
) -> str:
    goals_a, goals_b = parse_score(score)
    if total_goal_bucket(goals_a + goals_b) in bucket_options:
        return score
    raw_a, raw_b = parse_score(raw_score)
    bucket = choose_market_bucket_from_options(raw_a + raw_b, bucket_options, total_probabilities)
    return market_score_to_bucket(raw_score, bucket, total_probabilities)


def market_score_constrained_to_primary_bucket(
    raw_score: str,
    selected_bucket: str,
    total_probabilities: dict[int, float],
) -> str:
    return market_score_to_bucket(
        raw_score,
        selected_bucket,
        total_probabilities,
        MARKET_SCORE_MARGIN_DISCOUNT,
    )


def source_lineup_score_confidence(
    context_a: TeamContext | None,
    context_b: TeamContext | None,
) -> float:
    if context_a is None or context_b is None:
        return 1.0
    return (
        context_a.source_confidence_multiplier
        + context_b.source_confidence_multiplier
        + context_a.lineup_certainty_multiplier
        + context_b.lineup_certainty_multiplier
    ) / 4.0


def choose_realtime_second_bucket(
    total_buckets: list[tuple[str, float]],
    selected_bucket: str,
    raw_market_a: int,
    raw_market_b: int,
    shape_labels: str,
    recent_goal_signal: RecentGoalSignal,
) -> str:
    labels = {label for label in shape_labels.split(";") if label}
    if "credible_opponent" in labels and selected_bucket == "4-5球" and recent_goal_signal.has_big_goal:
        for bucket, _ in total_buckets:
            if bucket == "6-8球":
                return bucket
    return choose_second_total_goal_bucket(
        total_buckets,
        selected_bucket,
        raw_market_a,
        raw_market_b,
        shape_labels,
    )


def parse_total_goal_buckets(value: str) -> list[tuple[str, float]]:
    buckets: list[tuple[str, float]] = []
    for part in value.split(";"):
        pieces = part.strip().split()
        if len(pieces) < 2:
            continue
        buckets.append((pieces[0], float(pieces[1].removesuffix("%")) / 100.0))
    return buckets


def normalize_total_goal_buckets(buckets: dict[str, float]) -> list[tuple[str, float]]:
    total = sum(buckets.values())
    if total <= 0:
        raise RuntimeError("realtime total-goal buckets sum to zero")
    return sorted(((bucket, value / total) for bucket, value in buckets.items()), key=lambda item: item[1], reverse=True)


def multiply_realtime_bucket(buckets: dict[str, float], bucket: str, factor: float) -> None:
    damped_factor = 1.0 + (factor - 1.0) * REALTIME_TOTAL_GOAL_BUCKET_ADJUSTMENT_STRENGTH
    buckets[bucket] *= damped_factor


def realtime_total_goal_buckets(
    row: dict,
    base_buckets: list[tuple[str, float]],
    shape_labels: str,
    group_context: GroupStageContext,
    pair_tempo: float,
    p_draw: float,
    context_a: TeamContext | None,
    context_b: TeamContext | None,
) -> list[tuple[str, float]]:
    buckets = {bucket: probability for bucket, probability in base_buckets}
    labels = {label for label in shape_labels.split(";") if label}
    controlled_favorite = "controlled_favorite" in labels

    if "low_block" in labels or "low_event_favorite" in labels:
        multiply_realtime_bucket(buckets, "0-1球", 1.24)
        multiply_realtime_bucket(buckets, "4-5球", 0.80)
        multiply_realtime_bucket(buckets, "6-8球", 0.72)
    if ("transition_dog" in labels or "set_piece_risk" in labels) and not controlled_favorite:
        multiply_realtime_bucket(buckets, "2-3球", 1.06)
        multiply_realtime_bucket(buckets, "4-5球", 1.10)
        multiply_realtime_bucket(buckets, "0-1球", 0.92)
    if controlled_favorite:
        multiply_realtime_bucket(buckets, "0-1球", 1.16)
        multiply_realtime_bucket(buckets, "4-5球", 0.86)
        multiply_realtime_bucket(buckets, "6-8球", 0.72)
    if ("open_game" in labels or "open_mismatch" in labels) and not controlled_favorite:
        multiply_realtime_bucket(buckets, "0-1球", 0.74)
        multiply_realtime_bucket(buckets, "4-5球", 1.22)
        multiply_realtime_bucket(buckets, "6-8球", 1.24)
    if "collapse_risk" in labels:
        multiply_realtime_bucket(buckets, "0-1球", 0.60)
        multiply_realtime_bucket(buckets, "4-5球", 1.20)
        multiply_realtime_bucket(buckets, "6-8球", 1.46)

    if group_context.tempo_multiplier < 0.96:
        multiply_realtime_bucket(buckets, "0-1球", 1.10)
        multiply_realtime_bucket(buckets, "4-5球", 0.92)
        multiply_realtime_bucket(buckets, "6-8球", 0.86)
    elif group_context.tempo_multiplier > 1.04:
        multiply_realtime_bucket(buckets, "0-1球", 0.90)
        multiply_realtime_bucket(buckets, "4-5球", 1.10)
        multiply_realtime_bucket(buckets, "6-8球", 1.10)

    if pair_tempo < 0.92:
        multiply_realtime_bucket(buckets, "0-1球", 1.14)
        multiply_realtime_bucket(buckets, "4-5球", 0.88)
        multiply_realtime_bucket(buckets, "6-8球", 0.80)
    elif pair_tempo > 1.08:
        multiply_realtime_bucket(buckets, "0-1球", 0.86)
        multiply_realtime_bucket(buckets, "4-5球", 1.12)
        multiply_realtime_bucket(buckets, "6-8球", 1.12)

    if p_draw >= 0.34:
        multiply_realtime_bucket(buckets, "0-1球", 1.12)
        multiply_realtime_bucket(buckets, "6-8球", 0.84)

    if context_a is not None and context_b is not None:
        max_temperature = max(float(context_a.weather_high_c or 0), float(context_b.weather_high_c or 0))
        if max_temperature >= 33.0 and (context_a.weather_multiplier <= 0.95 or context_b.weather_multiplier <= 0.95):
            multiply_realtime_bucket(buckets, "0-1球", 1.12)
            multiply_realtime_bucket(buckets, "4-5球", 0.90)
            multiply_realtime_bucket(buckets, "6-8球", 0.84)

    return normalize_total_goal_buckets(buckets)


def select_aggressive_realtime_score(
    cells: list[tuple[int, int, float]],
    total_goal_values: set[int],
    predicted_outcome: str,
    p_a: float,
    p_draw: float,
    p_b: float,
    lambda_a: float,
    lambda_b: float,
    excluded_scores: set[tuple[int, int]],
    shape_labels: str,
    aggressive_score_bucket: str,
    score_confidence: float = 1.0,
) -> tuple[int, int, float]:
    labels = {label for label in shape_labels.split(";") if label}
    if score_confidence >= AGGRESSIVE_SCORE_CONFIDENCE_THRESHOLD:
        probabilities = total_goal_probability_lookup(cells)
        valid_totals = [
            total
            for total in sorted(total_goal_values_for_bucket(aggressive_score_bucket))
            if total in total_goal_values
            and probabilities.get(total, 0.0) > 0
            and (
                (predicted_outcome == "D" and total % 2 == 0)
                or (predicted_outcome != "D" and total > 0)
            )
        ]
        if valid_totals:
            best_probability = max(probabilities.get(total, 0.0) for total in valid_totals)
            close_totals = [
                total
                for total in valid_totals
                if probabilities.get(total, 0.0) >= best_probability * 0.75
            ]
            total_goal_values = {max(close_totals)}

    if "credible_opponent" not in labels or aggressive_score_bucket != "6-8球":
        return select_score_by_total_and_margin(
            cells,
            total_goal_values,
            predicted_outcome,
            p_a,
            p_draw,
            p_b,
            lambda_a,
            lambda_b,
            excluded_scores,
        )

    candidates = [
        cell
        for cell in cells
        if (cell[0], cell[1]) not in excluded_scores
        and cell[0] + cell[1] in total_goal_values
        and score_matches_outcome(cell[0], cell[1], predicted_outcome)
        and min(cell[0], cell[1]) >= 2
    ]
    if candidates:
        return min(
            candidates,
            key=lambda cell: (
                abs((cell[0] + cell[1]) - 6),
                abs(cell[0] - cell[1]),
                -cell[2],
            ),
        )
    return select_score_by_total_and_margin(
        cells,
        total_goal_values,
        predicted_outcome,
        p_a,
        p_draw,
        p_b,
        lambda_a,
        lambda_b,
        excluded_scores,
    )


def select_backup_score_inside_bucket(
    cells: list[tuple[int, int, float]],
    bucket: str,
    predicted_outcome: str,
    excluded_scores: set[tuple[int, int]],
) -> tuple[int, int, float]:
    candidates = [
        cell
        for cell in cells
        if (cell[0], cell[1]) not in excluded_scores
        and total_goal_bucket(cell[0] + cell[1]) == bucket
        and score_outcome(cell[0], cell[1]) == predicted_outcome
    ]
    if candidates:
        return candidates[0]

    fallback_candidates = [
        cell
        for cell in cells
        if (cell[0], cell[1]) not in excluded_scores
        and total_goal_bucket(cell[0] + cell[1]) == bucket
    ]
    if fallback_candidates:
        return fallback_candidates[0]
    raise RuntimeError(f"no backup score found inside bucket={bucket}")


def ensure_backup_score_inside_bucket(
    cells: list[tuple[int, int, float]],
    bucket: str,
    predicted_outcome: str,
    score: tuple[int, int, float],
    excluded_scores: set[tuple[int, int]],
) -> tuple[int, int, float]:
    if (
        (score[0], score[1]) not in excluded_scores
        and total_goal_bucket(score[0] + score[1]) == bucket
    ):
        return score
    return select_backup_score_inside_bucket(cells, bucket, predicted_outcome, excluded_scores)


def select_realtime_upset_score(
    cells: list[tuple[int, int, float]],
    upset_buckets: set[str],
    model_score: tuple[int, int, float],
    excluded_scores: set[tuple[int, int]],
    shape_labels: str,
    recent_goal_signal: RecentGoalSignal,
    p_a: float,
    p_b: float,
) -> tuple[int, int, float]:
    del shape_labels, recent_goal_signal
    model_outcome = score_outcome(model_score[0], model_score[1])
    favorite_outcome = "A" if p_a >= p_b else "B"
    underdog_outcome = "B" if favorite_outcome == "A" else "A"
    allowed_outcomes = {"D", underdog_outcome} - {model_outcome}
    if not allowed_outcomes:
        allowed_outcomes = {underdog_outcome}
    candidates = [
        cell
        for cell in cells
        if (cell[0], cell[1]) not in excluded_scores
        and total_goal_bucket(cell[0] + cell[1]) in upset_buckets
        and score_outcome(cell[0], cell[1]) in allowed_outcomes
    ]
    if candidates:
        return candidates[0]

    fallback_candidates = [
        cell
        for cell in cells
        if (cell[0], cell[1]) not in excluded_scores
        and score_outcome(cell[0], cell[1]) in allowed_outcomes
    ]
    if fallback_candidates:
        return fallback_candidates[0]
    raise RuntimeError(f"no realtime upset score found for outcomes={allowed_outcomes}")


def ensure_realtime_upset_direction(
    cells: list[tuple[int, int, float]],
    selected_bucket: str,
    complement_bucket: str,
    model_score: tuple[int, int, float],
    backup_score: tuple[int, int, float],
    market_score: str,
    upset_score: tuple[int, int, float],
    p_a: float,
    p_b: float,
) -> tuple[int, int, float]:
    model_outcome = score_outcome(model_score[0], model_score[1])
    favorite_outcome = "A" if p_a >= p_b else "B"
    underdog_outcome = "B" if favorite_outcome == "A" else "A"
    allowed_outcomes = {"D", underdog_outcome} - {model_outcome}
    if not allowed_outcomes:
        allowed_outcomes = {underdog_outcome}
    if score_outcome(upset_score[0], upset_score[1]) in allowed_outcomes:
        return upset_score

    market_a, market_b = parse_score(market_score)
    excluded_scores = {
        (model_score[0], model_score[1]),
        (backup_score[0], backup_score[1]),
        (market_a, market_b),
    }
    upset_buckets = upset_total_goal_buckets(selected_bucket, complement_bucket, model_score)
    candidates = [
        cell
        for cell in cells
        if (cell[0], cell[1]) not in excluded_scores
        and total_goal_bucket(cell[0] + cell[1]) in upset_buckets
        and score_outcome(cell[0], cell[1]) in allowed_outcomes
    ]
    if not candidates:
        candidates = [
            cell
            for cell in cells
            if (cell[0], cell[1]) not in excluded_scores
            and score_outcome(cell[0], cell[1]) in allowed_outcomes
        ]
    if not candidates:
        raise RuntimeError(f"no corrected upset score found for outcomes={allowed_outcomes}")

    return min(
        candidates,
        key=lambda cell: (
            0 if score_outcome(cell[0], cell[1]) == "D" else 1,
            cell[0] + cell[1],
            abs(cell[0] - cell[1]),
            -cell[2],
        ),
    )


def score_cell_for_pair(cells: list[tuple[int, int, float]], goals_a: int, goals_b: int) -> tuple[int, int, float]:
    for cell in cells:
        if cell[0] == goals_a and cell[1] == goals_b:
            return cell
    raise RuntimeError(f"score cell not found: {goals_a}-{goals_b}")


def score_for_outcome_in_bucket(
    selected_bucket: str,
    target_outcome: str,
    cells: list[tuple[int, int, float]],
) -> tuple[int, int, float]:
    candidates = [
        cell
        for cell in cells
        if total_goal_bucket(cell[0] + cell[1]) == selected_bucket
        and score_outcome(cell[0], cell[1]) == target_outcome
    ]
    if not candidates:
        raise RuntimeError(f"no score for outcome={target_outcome} in bucket={selected_bucket}")
    return candidates[0]


def resolve_model_score_outcome_conflict(
    selected_bucket: str,
    predicted_outcome: str,
    score: tuple[int, int, float],
    cells: list[tuple[int, int, float]],
    p_a: float,
    p_draw: float,
    p_b: float,
) -> tuple[int, int, float]:
    score_direction = score_outcome(score[0], score[1])
    if score_direction == predicted_outcome:
        return score

    if predicted_outcome == "D":
        return score_for_outcome_in_bucket(selected_bucket, "D", cells)

    if score_direction == "D":
        top_non_draw = max(p_a, p_b)
        if (
            p_draw >= MODEL_SCORE_DRAW_AGAINST_WIN_MIN_DRAW_PROBABILITY
            and top_non_draw - p_draw <= MODEL_SCORE_DRAW_AGAINST_WIN_MAX_EDGE
        ):
            return score
        return score_for_outcome_in_bucket(selected_bucket, predicted_outcome, cells)

    return score_for_outcome_in_bucket(selected_bucket, predicted_outcome, cells)


def early_knockout_score_ladder(
    stage: str,
    selected_bucket: str,
    score_1: tuple[int, int, float],
    score_2: tuple[int, int, float],
    market_score: str,
    score_4: tuple[int, int, float],
    cells: list[tuple[int, int, float]],
    p_draw: float,
    lambda_a: float,
    lambda_b: float,
) -> tuple[tuple[int, int, float], tuple[int, int, float], str, tuple[int, int, float]]:
    if stage.upper() not in KNOCKOUT_EARLY_HIGH_BUCKET_CAP_STAGES:
        return score_1, score_2, market_score, score_4

    favorite_is_a = lambda_a >= lambda_b
    favorite_xg = max(lambda_a, lambda_b)
    underdog_xg = min(lambda_a, lambda_b)
    xg_gap = abs(lambda_a - lambda_b)
    if selected_bucket == "2-3球":
        if xg_gap < KNOCKOUT_SCORE_LADDER_CLOSE_XG_GAP and p_draw >= KNOCKOUT_SCORE_LADDER_CLOSE_DRAW_MIN_PROBABILITY:
            pairs = [(1, 1), (2, 1), (2, 0), (0, 1)] if favorite_is_a else [(1, 1), (1, 2), (0, 2), (1, 0)]
        elif (
            favorite_xg >= KNOCKOUT_SCORE_LADDER_STRONG_FAVORITE_XG
            and underdog_xg <= KNOCKOUT_SCORE_LADDER_STRONG_UNDERDOG_XG_MAX
            and xg_gap >= KNOCKOUT_SCORE_LADDER_STRONG_XG_GAP
            and p_draw < KNOCKOUT_SCORE_LADDER_STRONG_DRAW_MAX_PROBABILITY
        ):
            pairs = [(2, 0), (2, 1), (3, 0), (1, 1)] if favorite_is_a else [(0, 2), (1, 2), (0, 3), (1, 1)]
        else:
            pairs = [(2, 1), (2, 0), (1, 1), (3, 0)] if favorite_is_a else [(1, 2), (0, 2), (1, 1), (0, 3)]
    elif selected_bucket == "4-5球" and p_draw >= KNOCKOUT_SCORE_LADDER_HIGH_DRAW_MIN_PROBABILITY:
        pairs = [(2, 2), (3, 1), (3, 2), (1, 1)] if favorite_is_a else [(2, 2), (1, 3), (2, 3), (1, 1)]
    else:
        return score_1, score_2, market_score, score_4

    ladder = [score_cell_for_pair(cells, goals_a, goals_b) for goals_a, goals_b in pairs]
    return ladder[0], ladder[1], format_score(ladder[2]), ladder[3]


def top_three_total_goal_buckets(selected_bucket: str, complement_bucket: str) -> set[str]:
    buckets = [selected_bucket, complement_bucket]
    for bucket in TOTAL_GOAL_BUCKET_LABELS:
        if bucket not in buckets:
            buckets.append(bucket)
    return set(buckets[1:3])


def upset_total_goal_buckets(
    selected_bucket: str,
    complement_bucket: str,
    model_score: tuple[int, int, float],
) -> set[str]:
    if score_outcome(model_score[0], model_score[1]) == "D":
        return {selected_bucket, complement_bucket}
    return top_three_total_goal_buckets(selected_bucket, complement_bucket)


def second_bucket_from_expected_total_goals(
    expected_total_goals: float,
    selected_bucket: str,
    shape_labels: str = "",
) -> str:
    labels = {
        label
        for label in shape_labels.split(";")
        if label and label not in {"low_block", "low_event_favorite", "low_event"}
    }
    return profile_second_bucket_from_expected_total_goals(
        expected_total_goals,
        selected_bucket,
        ";".join(labels),
    )


@dataclass(frozen=True)
class RealtimeSelection:
    p_a: float
    p_draw: float
    p_b: float
    predicted_outcome: str
    selected_bucket: str
    complement_bucket: str
    top_two_total_goal_labels: str
    score_1: tuple[int, int, float]
    score_2: tuple[int, int, float]
    market_value_adjusted_score: str
    score_4: tuple[int, int, float]


def select_realtime_outputs(
    *,
    row: dict,
    kickoff_bjt: datetime,
    completed_matches: list[CompletedMatch],
    group_context: GroupStageContext,
    shape_labels: str,
    style_edge: float,
    rank_a: int,
    rank_b: int,
    lambda_a: float,
    lambda_b: float,
    p_a: float,
    p_draw: float,
    p_b: float,
    predicted_outcome: str,
    context_a: TeamContext | None,
    context_b: TeamContext | None,
) -> RealtimeSelection:
    conservative_cells = outcome_adjusted_scores(lambda_a, lambda_b, p_a, p_draw, p_b)
    expected_total_goals = expected_total_goals_value(
        lambda_a,
        lambda_b,
        p_a,
        p_draw,
        p_b,
        shape_labels=shape_labels,
    )
    expected_total_goals = group_draw_total_goal_adjustment(
        expected_total_goals,
        p_a,
        p_draw,
        p_b,
        shape_labels,
        group_context,
    )
    expected_total_goals += group_context.open_tail_total_goals
    total_buckets = total_goal_bucket_probabilities_from_expected(expected_total_goals)
    raw_market_score = row.get("market_value_raw_score") or row["market_value_score"]
    parse_score(raw_market_score)
    favorite_recent_goal_signal = recent_goal_signal(
        match_favorite_team(row, p_a, p_b),
        kickoff_bjt,
        completed_matches,
    )
    selected_bucket = constrained_realtime_total_goal_bucket(
        row["selected_total_goal_bucket"],
        total_buckets[0][0],
        shape_labels,
        group_context,
    )
    selected_bucket, total_buckets = protect_strong_favorite_from_low_bucket(
        selected_bucket,
        total_buckets,
        lambda_a,
        lambda_b,
        p_a,
        p_b,
    )
    selected_bucket, total_buckets, high_bucket_capped = apply_knockout_high_bucket_cap(
        row["group"],
        selected_bucket,
        total_buckets,
        predicted_outcome,
        lambda_a,
        lambda_b,
        p_a,
        p_draw,
        p_b,
        shape_labels,
        group_context,
    )
    p_a, p_draw, p_b, predicted_outcome = apply_qualified_low_bucket_outcome_adjustment(
        p_a,
        p_draw,
        p_b,
        predicted_outcome,
        selected_bucket,
        group_context,
    )
    predicted_outcome = apply_knockout_draw_score_adjustment(
        p_a,
        p_draw,
        p_b,
        predicted_outcome,
        selected_bucket,
        group_context,
        shape_labels,
        lambda_a,
        lambda_b,
    )
    predicted_outcome = apply_style_matchup_outcome_adjustment(
        predicted_outcome,
        style_edge,
        p_a,
        p_draw,
        p_b,
        lambda_a,
        lambda_b,
        shape_labels,
    )
    conservative_cells = outcome_adjusted_scores(lambda_a, lambda_b, p_a, p_draw, p_b)
    complement_bucket = (
        next(bucket for bucket, _ in total_buckets if bucket != selected_bucket)
        if high_bucket_capped
        else second_bucket_from_expected_total_goals(expected_total_goals, selected_bucket, shape_labels)
    )
    selected_total_goal = choose_total_goal_in_bucket(
        conservative_cells,
        selected_bucket,
        predicted_outcome,
        p_a,
        p_b,
        lambda_a,
        lambda_b,
    )
    try:
        score_1 = select_score_by_total_and_margin(
            conservative_cells,
            {selected_total_goal},
            predicted_outcome,
            p_a,
            p_draw,
            p_b,
            lambda_a,
            lambda_b,
        )
    except RuntimeError:
        score_1, _, _ = select_recommended_score(conservative_cells, predicted_outcome)

    aggressive_lambda_a, aggressive_lambda_b = aggressive_score_lambdas(rank_a, rank_b, lambda_a, lambda_b)
    aggressive_cells = outcome_adjusted_scores(aggressive_lambda_a, aggressive_lambda_b, p_a, p_draw, p_b)
    aggressive_score_total = choose_total_goal_in_bucket(
        aggressive_cells,
        complement_bucket,
        predicted_outcome,
        p_a,
        p_b,
        aggressive_lambda_a,
        aggressive_lambda_b,
    )
    excluded_scores = {(score_1[0], score_1[1])}
    try:
        score_2 = select_aggressive_realtime_score(
            aggressive_cells,
            {aggressive_score_total},
            predicted_outcome,
            p_a,
            p_draw,
            p_b,
            aggressive_lambda_a,
            aggressive_lambda_b,
            excluded_scores,
            shape_labels,
            complement_bucket,
            source_lineup_score_confidence(context_a, context_b),
        )
    except RuntimeError:
        try:
            score_2 = select_backup_score_inside_bucket(
                aggressive_cells,
                complement_bucket,
                predicted_outcome,
                excluded_scores,
            )
        except RuntimeError:
            if context_a is None or context_b is None:
                raise
            score_2, _, _ = select_recommended_score(aggressive_cells, predicted_outcome)

    market_score_input = (
        raw_market_score
        if context_a is None or context_b is None
        else market_value_micro_adjust(raw_market_score, context_a, context_b)
    )
    market_value_adjusted_score = market_score_constrained_to_primary_bucket(
        market_score_input,
        selected_bucket,
        total_goal_probability_lookup(conservative_cells),
    )
    market_a, market_b = parse_score(market_value_adjusted_score)
    score_4 = select_realtime_upset_score(
        conservative_cells,
        upset_total_goal_buckets(selected_bucket, complement_bucket, score_1),
        score_1,
        {
            (score_1[0], score_1[1]),
            (score_2[0], score_2[1]),
            (market_a, market_b),
        },
        shape_labels,
        favorite_recent_goal_signal,
        p_a,
        p_b,
    )
    score_1, score_2, market_value_adjusted_score, score_4 = early_knockout_score_ladder(
        row["group"],
        selected_bucket,
        score_1,
        score_2,
        market_value_adjusted_score,
        score_4,
        conservative_cells,
        p_draw,
        lambda_a,
        lambda_b,
    )
    score_1 = resolve_model_score_outcome_conflict(
        selected_bucket,
        predicted_outcome,
        score_1,
        conservative_cells,
        p_a,
        p_draw,
        p_b,
    )
    score_2 = ensure_backup_score_inside_bucket(
        conservative_cells,
        complement_bucket,
        predicted_outcome,
        score_2,
        {(score_1[0], score_1[1])},
    )
    score_4 = ensure_realtime_upset_direction(
        conservative_cells,
        selected_bucket,
        complement_bucket,
        score_1,
        score_2,
        market_value_adjusted_score,
        score_4,
        p_a,
        p_b,
    )
    return RealtimeSelection(
        p_a=p_a,
        p_draw=p_draw,
        p_b=p_b,
        predicted_outcome=predicted_outcome,
        selected_bucket=selected_bucket,
        complement_bucket=complement_bucket,
        top_two_total_goal_labels=format_total_goal_buckets(total_buckets, selected_bucket, complement_bucket),
        score_1=score_1,
        score_2=score_2,
        market_value_adjusted_score=market_value_adjusted_score,
        score_4=score_4,
    )


def build_adjusted_output_row(
    *,
    row: dict,
    selection: RealtimeSelection,
    lambda_a: float,
    lambda_b: float,
    shape: MatchShapeContext | None,
    shape_labels: str,
    style_features_a: frozenset[str],
    style_features_b: frozenset[str],
    style_effect: StyleMatchupEffect,
    style_influence: float,
    profile_a: TeamShapeProfile | None,
    profile_b: TeamShapeProfile | None,
    group_context: GroupStageContext,
    context_a: TeamContext | None,
    context_b: TeamContext | None,
    context_chain_multipliers: str,
    context_signal_tags: str,
    key_player_signal_tags: str,
) -> dict:
    context_applied = context_a is not None and context_b is not None
    notes = (
        f"{row['team_a']}: {context_a.notes} | {row['team_b']}: {context_b.notes}"
        if context_applied
        else ""
    )
    sources = (
        f"{row['team_a']}: {context_a.source_urls} | {row['team_b']}: {context_b.source_urls}"
        if context_applied
        else ""
    )
    risk_label, risk_reasons = adjusted_risk_fields(row, group_context)
    return {
        **row,
        "predicted_outcome": selection.predicted_outcome,
        "risk_label": risk_label,
        "risk_reasons": risk_reasons,
        "context_applied": "TRUE" if context_applied else "FALSE",
        "shape_applied": "TRUE" if shape is not None else "FALSE",
        "shape_labels": shape_labels,
        "shape_notes": shape.notes if shape is not None else "",
        "style_features_a": ";".join(sorted(style_features_a)),
        "style_features_b": ";".join(sorted(style_features_b)),
        "style_matchup_edge": f"{style_effect.edge:.6f}",
        "style_matchup_influence": f"{style_influence:.3f}",
        "style_matchup_points_shift": f"{style_effect.points_shift:.3f}",
        "style_matchup_total_multiplier": f"{style_effect.total_goal_multiplier:.4f}",
        "style_matchup_reasons": "; ".join(style_effect.reasons),
        "team_shape_labels_a": ";".join(sorted(profile_a.derived_labels)) if profile_a else "",
        "team_shape_labels_b": ";".join(sorted(profile_b.derived_labels)) if profile_b else "",
        "team_shape_reason_a": profile_a.reason if profile_a else "",
        "team_shape_reason_b": profile_b.reason if profile_b else "",
        "team_shape_profile_mode": TEAM_SHAPE_PROFILE_MODE,
        "group_round": str(group_context.round_number),
        "draw_acceptance_a": f"{group_context.draw_acceptance_a:.2f}",
        "draw_acceptance_b": f"{group_context.draw_acceptance_b:.2f}",
        "group_draw_multiplier": f"{group_context.draw_multiplier:.2f}",
        "group_tempo_multiplier": f"{group_context.tempo_multiplier:.2f}",
        "group_context_complete": "TRUE" if group_context.complete else "FALSE",
        "group_context_notes": group_context.notes,
        "adjusted_p_a": f"{selection.p_a:.6f}",
        "adjusted_p_draw": f"{selection.p_draw:.6f}",
        "adjusted_p_b": f"{selection.p_b:.6f}",
        "adjusted_xg_a": f"{lambda_a:.4f}",
        "adjusted_xg_b": f"{lambda_b:.4f}",
        "adjusted_total_goal_bucket": selection.selected_bucket,
        "backup_total_goal_bucket": selection.complement_bucket,
        "adjusted_total_goals_top2": selection.top_two_total_goal_labels,
        "bucket_primary_score": format_score(selection.score_1),
        "adjusted_score_1_model": format_score(selection.score_1),
        "bucket_complement_score": format_score(selection.score_2),
        "aggressive_score": format_score(selection.score_2),
        "adjusted_score_2_aggressive_prediction": format_score(selection.score_2),
        "adjusted_score_3_market_value": selection.market_value_adjusted_score,
        "upset_score": format_score(selection.score_4),
        "upset_score_probability": f"{selection.score_4[2]:.6f}",
        "adjusted_score_4_upset": format_score(selection.score_4),
        "adjusted_score_4_upset_probability": f"{selection.score_4[2]:.6f}",
        "xg_goal_diff": f"{lambda_a - lambda_b:.4f}",
        "xg_outcome_edge": f"{xg_outcome_probability_edge(lambda_a, lambda_b, selection.predicted_outcome):.6f}",
        "legacy_outcome_edge": f"{legacy_outcome_probability_edge(selection.p_a, selection.p_draw, selection.p_b, selection.predicted_outcome):.6f}",
        "outcome_edge_conflict": (
            "TRUE"
            if outcome_edge_conflict(
                lambda_a,
                lambda_b,
                selection.p_a,
                selection.p_draw,
                selection.p_b,
                selection.predicted_outcome,
            )
            else "FALSE"
        ),
        "context_chain_multipliers": context_chain_multipliers,
        "context_signal_tags": ";".join(part for part in (context_signal_tags, key_player_signal_tags) if part),
        "source_confidence_a": f"{context_a.source_confidence_multiplier:.2f}" if context_applied else "",
        "source_confidence_b": f"{context_b.source_confidence_multiplier:.2f}" if context_applied else "",
        "lineup_certainty_a": f"{context_a.lineup_certainty_multiplier:.2f}" if context_applied else "",
        "lineup_certainty_b": f"{context_b.lineup_certainty_multiplier:.2f}" if context_applied else "",
        "context_notes": notes,
        "context_sources": sources,
    }


def apply_context(
    row: dict,
    contexts: dict[tuple[str, str], TeamContext],
    shapes: dict[str, MatchShapeContext],
    team_shape_profiles: dict[str, list[TeamShapeProfile]],
    key_player_signals: dict[str, KeyPlayerSignal],
    key_player_statuses: dict[tuple[str, str, str], KeyPlayerMatchStatus],
    team_market_values: dict,
    completed_matches: list[CompletedMatch],
) -> dict:
    kickoff_bjt = parse_bjt_datetime(row["date_bjt"], row["time_bjt"])
    match_name = f"{row['team_a']} vs {row['team_b']}"
    match_key = f"{row['date_bjt']} {row['time_bjt']}"
    context_a = contexts.get((match_name, row["team_a"]))
    context_b = contexts.get((match_name, row["team_b"]))
    explicit_shape = shapes.get(match_name)
    p_a_base = float(row["p_a"])
    p_b_base = float(row["p_b"])
    profile_a = team_shape_profile_at(team_shape_profiles, row["team_a"], kickoff_bjt)
    profile_b = team_shape_profile_at(team_shape_profiles, row["team_b"], kickoff_bjt)
    profile_shape = inferred_team_shape_context(
        profile_a,
        profile_b,
        row["team_a"],
        row["team_b"],
        p_a_base,
        p_b_base,
    )
    shape = select_shape_context(explicit_shape, profile_shape)
    shape_labels = shape.pre_match_shapes if shape is not None else ""
    style_features_a = realtime_style_features(row, "a", profile_a)
    style_features_b = realtime_style_features(row, "b", profile_b)
    raw_style_effect = style_matchup_effect(style_features_a, style_features_b)
    style_influence = 1.0
    style_effect = raw_style_effect
    group_context = group_stage_context(row, completed_matches)
    rank_a = int(row["fifa_rank_a"])
    rank_b = int(row["fifa_rank_b"])
    context_chain_multipliers = ""
    context_signal_tags = ""
    if context_a is not None and context_b is not None:
        context_chain_multipliers = (
            f"{row['team_a']} a/d/t="
            f"{context_a.attack_multiplier:.2f}/{context_a.opponent_attack_multiplier:.2f}/{context_a.tempo_multiplier:.2f}"
            f", src/line={context_a.source_confidence_multiplier:.2f}/{context_a.lineup_certainty_multiplier:.2f}; "
            f"{row['team_b']} a/d/t="
            f"{context_b.attack_multiplier:.2f}/{context_b.opponent_attack_multiplier:.2f}/{context_b.tempo_multiplier:.2f}"
            f", src/line={context_b.source_confidence_multiplier:.2f}/{context_b.lineup_certainty_multiplier:.2f}"
        )
        tags = []
        if context_a.defense_leak_evidence:
            tags.append(f"{row['team_a']}:defense_leak")
        if context_a.underdog_goal_evidence:
            tags.append(f"{row['team_a']}:goal_route")
        if context_b.defense_leak_evidence:
            tags.append(f"{row['team_b']}:defense_leak")
        if context_b.underdog_goal_evidence:
            tags.append(f"{row['team_b']}:goal_route")
        context_signal_tags = ";".join(tags)
    key_player_effect_a = key_player_effect(
        row["team_a"],
        rank_a,
        match_key,
        match_name,
        context_a,
        key_player_signals,
        key_player_statuses,
        team_market_values,
    )
    key_player_effect_b = key_player_effect(
        row["team_b"],
        rank_b,
        match_key,
        match_name,
        context_b,
        key_player_signals,
        key_player_statuses,
        team_market_values,
    )
    key_player_signal_tags = ";".join(
        label
        for label in (
            f"{row['team_a']}:{key_player_effect_a.label}" if key_player_effect_a.label else "",
            f"{row['team_b']}:{key_player_effect_b.label}" if key_player_effect_b.label else "",
        )
        if label
    )
    if context_a is None or context_b is None:
        p_a = float(row["p_a"])
        p_draw = float(row["p_draw"])
        p_b = float(row["p_b"])
        p_a, p_draw, p_b = shape_adjusted_probabilities(p_a, p_draw, p_b, shape)
        p_a, p_draw, p_b = apply_group_stage_probabilities(p_a, p_draw, p_b, group_context)
        style_influence = style_influence_factor(
            p_a=p_a,
            p_b=p_b,
        )
        style_effect = apply_style_influence_gate(raw_style_effect, style_influence)
        lambda_a = (
            float(row["xg_a"])
            * group_context.tempo_multiplier
            * group_context.attack_multiplier_a
            * group_context.opponent_attack_multiplier_a
        )
        lambda_b = (
            float(row["xg_b"])
            * group_context.tempo_multiplier
            * group_context.attack_multiplier_b
            * group_context.opponent_attack_multiplier_b
        )
        lambda_a, lambda_b = apply_key_player_team_goals(
            lambda_a,
            lambda_b,
            key_player_effect_a,
            key_player_effect_b,
        )
        predicted_outcome = predicted_outcome_from_xg(lambda_a, lambda_b, group_context, shape_labels)
        lambda_a, lambda_b = apply_style_matchup_xg(
            lambda_a,
            lambda_b,
            style_effect.xg_scale_a,
            style_effect.xg_scale_b,
            style_effect.total_goal_multiplier,
        )
        selection = select_realtime_outputs(
            row=row,
            kickoff_bjt=kickoff_bjt,
            completed_matches=completed_matches,
            group_context=group_context,
            shape_labels=shape_labels,
            style_edge=style_effect.edge,
            rank_a=rank_a,
            rank_b=rank_b,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            p_a=p_a,
            p_draw=p_draw,
            p_b=p_b,
            predicted_outcome=predicted_outcome,
            context_a=context_a,
            context_b=context_b,
        )
        return build_adjusted_output_row(
            row=row,
            selection=selection,
            lambda_a=lambda_a,
            lambda_b=lambda_b,
            shape=shape,
            shape_labels=shape_labels,
            style_features_a=style_features_a,
            style_features_b=style_features_b,
            style_effect=style_effect,
            style_influence=style_influence,
            profile_a=profile_a,
            profile_b=profile_b,
            group_context=group_context,
            context_a=context_a,
            context_b=context_b,
            context_chain_multipliers=context_chain_multipliers,
            context_signal_tags=context_signal_tags,
            key_player_signal_tags=key_player_signal_tags,
        )

    lambda_a = float(row["xg_a"]) * context_a.attack_multiplier * context_b.opponent_attack_multiplier
    lambda_b = float(row["xg_b"]) * context_b.attack_multiplier * context_a.opponent_attack_multiplier
    p_a = float(row["p_a"])
    p_draw = float(row["p_draw"])
    p_b = float(row["p_b"])
    p_a, p_draw, p_b = shape_adjusted_probabilities(p_a, p_draw, p_b, shape)
    pair_tempo = context_a.tempo_multiplier * context_b.tempo_multiplier
    pair_tempo *= context_tempo_discount(context_a, context_b, p_a, p_draw, p_b)
    if shape is not None:
        pair_tempo *= shape.tempo_multiplier
    p_a, p_draw, p_b = apply_group_stage_probabilities(p_a, p_draw, p_b, group_context)
    style_influence = style_influence_factor(
        p_a=p_a,
        p_b=p_b,
    )
    style_effect = apply_style_influence_gate(raw_style_effect, style_influence)
    pair_tempo *= group_context.tempo_multiplier
    lambda_a *= pair_tempo
    lambda_b *= pair_tempo
    lambda_a *= group_context.attack_multiplier_a
    lambda_b *= group_context.attack_multiplier_b
    lambda_a *= group_context.opponent_attack_multiplier_a
    lambda_b *= group_context.opponent_attack_multiplier_b
    lambda_a, lambda_b = apply_shape_attack(lambda_a, lambda_b, p_a, p_b, shape)
    lambda_a, lambda_b = apply_key_player_team_goals(
        lambda_a,
        lambda_b,
        key_player_effect_a,
        key_player_effect_b,
    )
    predicted_outcome = predicted_outcome_from_xg(lambda_a, lambda_b, group_context, shape_labels)
    lambda_a, lambda_b = apply_style_matchup_xg(
        lambda_a,
        lambda_b,
        style_effect.xg_scale_a,
        style_effect.xg_scale_b,
        style_effect.total_goal_multiplier,
    )

    selection = select_realtime_outputs(
        row=row,
        kickoff_bjt=kickoff_bjt,
        completed_matches=completed_matches,
        group_context=group_context,
        shape_labels=shape_labels,
        style_edge=style_effect.edge,
        rank_a=rank_a,
        rank_b=rank_b,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        p_a=p_a,
        p_draw=p_draw,
        p_b=p_b,
        predicted_outcome=predicted_outcome,
        context_a=context_a,
        context_b=context_b,
    )
    return build_adjusted_output_row(
        row=row,
        selection=selection,
        lambda_a=lambda_a,
        lambda_b=lambda_b,
        shape=shape,
        shape_labels=shape_labels,
        style_features_a=style_features_a,
        style_features_b=style_features_b,
        style_effect=style_effect,
        style_influence=style_influence,
        profile_a=profile_a,
        profile_b=profile_b,
        group_context=group_context,
        context_a=context_a,
        context_b=context_b,
        context_chain_multipliers=context_chain_multipliers,
        context_signal_tags=context_signal_tags,
        key_player_signal_tags=key_player_signal_tags,
    )


def runtime_parameters() -> dict[str, object]:
    return {
        "team_shape_profile_mode": TEAM_SHAPE_PROFILE_MODE,
        "strong_favorite_low_bucket_min_favorite_xg": STRONG_FAVORITE_LOW_BUCKET_MIN_FAVORITE_XG,
        "strong_favorite_low_bucket_max_underdog_xg": STRONG_FAVORITE_LOW_BUCKET_MAX_UNDERDOG_XG,
        "strong_favorite_low_bucket_min_probability": STRONG_FAVORITE_LOW_BUCKET_MIN_PROBABILITY,
        "include_completed_match_keys": sorted(
            item.strip()
            for item in os.environ.get("WC_INCLUDE_COMPLETED_MATCH_KEYS", "").split(",")
            if item.strip()
        ),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    contexts = load_context()
    shapes = load_match_shapes()
    team_shape_profiles = load_team_shape_profiles()
    key_player_signals = load_key_player_signals()
    key_player_statuses = load_key_player_match_statuses()
    team_market_values = load_market_values()
    completed_matches = load_completed_matches()
    rows = [
        apply_context(
            row,
            contexts,
            shapes,
            team_shape_profiles,
            key_player_signals,
            key_player_statuses,
            team_market_values,
            completed_matches,
        )
        for row in load_predictions()
    ]
    realtime_output.write_csv(ADJUSTED_CSV, rows)
    realtime_output.write_markdown(ADJUSTED_MD, rows)
    cache_dir = realtime_output.write_realtime_cache(
        rows,
        cache_dir=REALTIME_CACHE_DIR,
        source_files=[
            ("base_predictions", PREDICTIONS_CSV),
            ("realtime_team_context", CONTEXT_CSV),
            ("match_shape_context", MATCH_SHAPE_CSV),
            ("in_tournament_team_shape_profiles", TEAM_SHAPE_PROFILE_CSV),
            ("key_player_signals", KEY_PLAYER_SIGNAL_CSV),
            ("key_player_match_status", KEY_PLAYER_MATCH_STATUS_CSV),
            ("world_cup_2026_results", RESULTS_CSV),
            ("international_results", INTERNATIONAL_RESULTS_CSV),
        ],
        output_files=[
            ("realtime_context_adjusted_plan_csv", ADJUSTED_CSV),
            ("realtime_context_adjusted_plan_md", ADJUSTED_MD),
        ],
        runtime_parameters=runtime_parameters(),
    )
    print(f"CSV: {ADJUSTED_CSV}")
    print(f"Markdown: {ADJUSTED_MD}")
    print(f"Cache: {cache_dir}")
    for row in rows:
        if row["context_applied"] == "TRUE":
            print(
                f"{row['date_bjt']} {row['team_a']} vs {row['team_b']}: "
                f"{row['selected_total_goal_bucket']}->{row.get('adjusted_total_goals_top2', row['adjusted_total_goal_bucket'])}; "
                f"{row['recommended_score']}->{row['adjusted_score_1_model']}; "
                f"{row['aggressive_score']}->{row['adjusted_score_2_aggressive_prediction']}; "
                f"{row['market_value_score']}->{row['adjusted_score_3_market_value']}; "
                f"{row.get('upset_score', '')}->{row.get('adjusted_score_4_upset', '')}"
            )


if __name__ == "__main__":
    main()
