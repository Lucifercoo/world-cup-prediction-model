from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from datetime import datetime

from predict import DATA_DIR, canonical_team


RESULTS_CSV = DATA_DIR / "world_cup_2026_results.csv"
MATCH_SHAPE_CSV = DATA_DIR / "match_shape_context.csv"
EVENTS_CSV = DATA_DIR / "in_tournament_adjustment_events.csv"
OUTPUT_CSV = DATA_DIR / "in_tournament_team_shape_profiles.csv"

LOW_EVENT_LABELS = {
    "low_block",
    "low_event",
    "low_event_favorite",
    "low_conversion",
    "controlled_favorite",
    "controlled_mismatch",
    "underdog_draw",
}
OPEN_EVENT_LABELS = {
    "open_game",
    "open_favorite",
    "open_mismatch",
    "collapse_risk",
    "red_card_collapse",
    "home_pressure",
    "early_goal",
}
CONTROL_LABELS = {"controlled_favorite", "controlled_mismatch", "favorite_win", "home_pressure"}
TRANSITION_LABELS = {"transition_dog", "set_piece_risk", "underdog_draw", "underdog_win"}
COLLAPSE_LABELS = {"collapse_risk", "red_card_collapse"}
ABNORMAL_LOW_WEIGHT_LABELS = {
    "red_card_collapse",
    "red_card_event",
    "late_penalty",
    "referee_lenient_anomaly",
    "officiating_anomaly",
}
STRONG_DEFENSE_REFERENCE_TEAMS = {
    "Argentina",
    "Belgium",
    "Brazil",
    "England",
    "France",
    "Germany",
    "Netherlands",
    "Portugal",
    "Uruguay",
}


@dataclass(frozen=True)
class Result:
    date_bjt: str
    time_bjt: str
    group: str
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int


@dataclass(frozen=True)
class Event:
    p_for: float
    result_surprise: float
    attack_surprise: float
    defense_surprise: float
    total_surprise: float


@dataclass
class TeamShapeState:
    effective_after_bjt: datetime
    played: int = 0
    low_event_score: float = 0.0
    strong_defense_attack_suppression_score: float = 0.0
    open_event_score: float = 0.0
    control_score: float = 0.0
    transition_score: float = 0.0
    defense_resistance_score: float = 0.0
    defense_fragility_score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def result_datetime(result: Result) -> datetime:
    return datetime.strptime(f"{result.date_bjt} {result.time_bjt}", "%Y-%m-%d %H:%M")


def result_key(result: Result) -> str:
    return "|".join([result.date_bjt, result.time_bjt, result.team_a, result.team_b])


def load_results() -> list[Result]:
    with RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = [
            Result(
                date_bjt=row["date_bjt"],
                time_bjt=row["time_bjt"],
                group=row["group"],
                team_a=canonical_team(row["team_a"]),
                team_b=canonical_team(row["team_b"]),
                goals_a=int(row["goals_a"]),
                goals_b=int(row["goals_b"]),
            )
            for row in csv.DictReader(fh)
        ]
    return sorted(rows, key=result_datetime)


def load_observed_shapes() -> dict[str, frozenset[str]]:
    if not MATCH_SHAPE_CSV.exists():
        raise RuntimeError(f"missing match shape file: {MATCH_SHAPE_CSV}")
    shapes: dict[str, frozenset[str]] = {}
    with MATCH_SHAPE_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            observed = frozenset(label for label in row["observed_shapes"].split(";") if label)
            shapes[row["match"]] = observed
    return shapes


def load_events() -> dict[tuple[str, str], Event]:
    if not EVENTS_CSV.exists():
        raise RuntimeError(f"missing adjustment events file: {EVENTS_CSV}")
    events: dict[tuple[str, str], Event] = {}
    with EVENTS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            events[(row["match_key"], canonical_team(row["team"]))] = Event(
                p_for=float(row["p_for"]),
                result_surprise=float(row["result_surprise"]),
                attack_surprise=float(row["attack_surprise"]),
                defense_surprise=float(row["defense_surprise"]),
                total_surprise=float(row["total_surprise"]),
            )
    return events


def abnormal_weight(event: Event, labels: frozenset[str]) -> float:
    surprise = (
        abs(event.result_surprise) * 0.90
        + min(2.5, abs(event.attack_surprise)) * 0.12
        + min(2.5, abs(event.defense_surprise)) * 0.10
        + min(3.0, abs(event.total_surprise)) * 0.10
    )
    weight = clamp(0.70 + surprise, 0.65, 1.55)
    if labels & ABNORMAL_LOW_WEIGHT_LABELS:
        weight *= 0.55
    return weight


def score_implied_labels(result: Result, event_a: Event, event_b: Event) -> frozenset[str]:
    labels: set[str] = set()
    total_goals = result.goals_a + result.goals_b
    if total_goals <= 1:
        labels.add("low_event")
        if min(event_a.attack_surprise, event_b.attack_surprise) < -0.70:
            labels.add("low_conversion")
    if total_goals >= 4:
        labels.add("open_game")

    if result.goals_a > result.goals_b:
        winner_event, loser_event = event_a, event_b
        loser_goals_against = result.goals_a
    elif result.goals_b > result.goals_a:
        winner_event, loser_event = event_b, event_a
        loser_goals_against = result.goals_b
    else:
        winner_event = loser_event = None
        loser_goals_against = 0

    if loser_event is not None and loser_goals_against >= 3 and loser_event.defense_surprise < -0.60:
        labels.add("collapse_risk")
    if winner_event is not None and winner_event.p_for >= 0.58 and total_goals <= 3:
        labels.add("controlled_favorite")
    if event_a.p_for < 0.30 and result.goals_a > 0 and event_a.attack_surprise > 0.15:
        labels.add("transition_dog")
    if event_b.p_for < 0.30 and result.goals_b > 0 and event_b.attack_surprise > 0.15:
        labels.add("transition_dog")
    return frozenset(labels)


def append_reason(state: TeamShapeState, reason: str) -> None:
    state.reasons.append(reason)
    if len(state.reasons) > 4:
        state.reasons = state.reasons[-4:]


def update_state(
    state: TeamShapeState,
    *,
    team: str,
    opponent: str,
    goals_for: int,
    goals_against: int,
    is_favorite: bool,
    event: Event,
    labels: frozenset[str],
    effective_after: datetime,
) -> None:
    weight = abnormal_weight(event, labels)
    state.effective_after_bjt = effective_after
    state.played += 1

    if labels & LOW_EVENT_LABELS:
        state.low_event_score += weight
    opponent_is_strong_or_credible = opponent in STRONG_DEFENSE_REFERENCE_TEAMS or event.p_for <= 0.70
    if (
        is_favorite
        and labels & {"low_event", "low_conversion", "controlled_favorite"}
        and opponent_is_strong_or_credible
        and goals_for <= 1
    ):
        state.strong_defense_attack_suppression_score += weight
    if labels & OPEN_EVENT_LABELS or goals_for + goals_against >= 4:
        state.open_event_score += weight
    if is_favorite and labels & CONTROL_LABELS:
        state.control_score += weight
    if (not is_favorite) and (labels & TRANSITION_LABELS) and (goals_for > 0 or event.attack_surprise > 0.15):
        state.transition_score += weight
    if goals_against == 0 and event.defense_surprise > 0.45:
        state.defense_resistance_score += weight
    if (not is_favorite) and (labels & LOW_EVENT_LABELS) and goals_against <= 1:
        state.defense_resistance_score += weight * 0.70
    if event.defense_surprise < -0.70 or ((labels & COLLAPSE_LABELS) and goals_against >= 3):
        state.defense_fragility_score += weight

    label_text = ";".join(sorted(labels)) if labels else "score_surprise"
    append_reason(
        state,
        (
            f"{effective_after.strftime('%m-%d')} {team} {goals_for}-{goals_against} {opponent}: "
            f"{label_text}, 权重{weight:.2f}"
        ),
    )


def ratio(score: float, played: int) -> float:
    if played <= 0:
        return 0.0
    return score / played


def derived_labels(state: TeamShapeState) -> list[str]:
    low = ratio(state.low_event_score, state.played)
    open_event = ratio(state.open_event_score, state.played)
    strong_defense_suppression = ratio(state.strong_defense_attack_suppression_score, state.played)
    control = ratio(state.control_score, state.played)
    transition = ratio(state.transition_score, state.played)
    resistance = ratio(state.defense_resistance_score, state.played)
    fragility = ratio(state.defense_fragility_score, state.played)
    labels: list[str] = []
    if low >= 0.45:
        labels.append("low_event_team")
    if strong_defense_suppression >= 0.35:
        labels.append("strong_defense_attack_suppression_team")
    if open_event >= 0.55 and open_event > low + 0.20:
        labels.append("open_event_team")
    if control >= 0.35:
        labels.append("control_team")
    if transition >= 0.30:
        labels.append("transition_route_team")
    if resistance >= 0.35:
        labels.append("defensive_resistance_team")
    if fragility >= 0.35:
        labels.append("defensive_fragility_team")
    return labels


def state_row(team: str, state: TeamShapeState) -> dict[str, str]:
    low = ratio(state.low_event_score, state.played)
    open_event = ratio(state.open_event_score, state.played)
    strong_defense_suppression = ratio(state.strong_defense_attack_suppression_score, state.played)
    control = ratio(state.control_score, state.played)
    transition = ratio(state.transition_score, state.played)
    resistance = ratio(state.defense_resistance_score, state.played)
    fragility = ratio(state.defense_fragility_score, state.played)
    draw_multiplier = clamp(1.0 + low * 0.05 + resistance * 0.04 - open_event * 0.04 - fragility * 0.03, 0.94, 1.08)
    tempo_multiplier = clamp(1.0 - low * 0.05 - control * 0.04 + open_event * 0.05 + fragility * 0.04, 0.93, 1.07)
    attack_multiplier = clamp(1.0 + transition * 0.04 - low * 0.03 - strong_defense_suppression * 0.04, 0.92, 1.06)
    opponent_attack_multiplier = clamp(1.0 - resistance * 0.04 + fragility * 0.05, 0.94, 1.07)
    return {
        "team": team,
        "effective_after_bjt": state.effective_after_bjt.strftime("%Y-%m-%d %H:%M"),
        "played": str(state.played),
        "low_event_score": f"{low:.3f}",
        "strong_defense_attack_suppression_score": f"{strong_defense_suppression:.3f}",
        "open_event_score": f"{open_event:.3f}",
        "control_score": f"{control:.3f}",
        "transition_score": f"{transition:.3f}",
        "defense_resistance_score": f"{resistance:.3f}",
        "defense_fragility_score": f"{fragility:.3f}",
        "draw_multiplier": f"{draw_multiplier:.3f}",
        "tempo_multiplier": f"{tempo_multiplier:.3f}",
        "attack_multiplier": f"{attack_multiplier:.3f}",
        "opponent_attack_multiplier": f"{opponent_attack_multiplier:.3f}",
        "derived_labels": ";".join(derived_labels(state)),
        "reason": " | ".join(state.reasons),
    }


def build_rows() -> list[dict[str, str]]:
    results = load_results()
    observed_shapes = load_observed_shapes()
    events = load_events()
    states: dict[str, TeamShapeState] = {}
    rows: list[dict[str, str]] = []
    for result in results:
        match_name = f"{result.team_a} vs {result.team_b}"
        key = result_key(result)
        event_a = events.get((key, result.team_a))
        event_b = events.get((key, result.team_b))
        if event_a is None or event_b is None:
            raise RuntimeError(f"missing adjustment event for {match_name}; run build_in_tournament_adjustments.py first")
        labels = frozenset(set(observed_shapes.get(match_name, frozenset())) | set(score_implied_labels(result, event_a, event_b)))
        effective_after = result_datetime(result)
        favorite_a = event_a.p_for >= event_b.p_for
        state_a = states.setdefault(result.team_a, TeamShapeState(effective_after_bjt=effective_after))
        state_b = states.setdefault(result.team_b, TeamShapeState(effective_after_bjt=effective_after))
        update_state(
            state_a,
            team=result.team_a,
            opponent=result.team_b,
            goals_for=result.goals_a,
            goals_against=result.goals_b,
            is_favorite=favorite_a,
            event=event_a,
            labels=labels,
            effective_after=effective_after,
        )
        update_state(
            state_b,
            team=result.team_b,
            opponent=result.team_a,
            goals_for=result.goals_b,
            goals_against=result.goals_a,
            is_favorite=not favorite_a,
            event=event_b,
            labels=labels,
            effective_after=effective_after,
        )
        rows.append(state_row(result.team_a, state_a))
        rows.append(state_row(result.team_b, state_b))
    return rows


def write_rows(rows: list[dict[str, str]]) -> None:
    fields = [
        "team",
        "effective_after_bjt",
        "played",
        "low_event_score",
        "strong_defense_attack_suppression_score",
        "open_event_score",
        "control_score",
        "transition_score",
        "defense_resistance_score",
        "defense_fragility_score",
        "draw_multiplier",
        "tempo_multiplier",
        "attack_multiplier",
        "opponent_attack_multiplier",
        "derived_labels",
        "reason",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    rows = build_rows()
    write_rows(rows)
    teams = sorted({row["team"] for row in rows})
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Teams: {len(teams)}")
    print(f"Snapshots: {len(rows)}")


if __name__ == "__main__":
    main()
