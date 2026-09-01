from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime

import predict_fifa_profile as pfp
from build_fifa_annual_rankings import normalized_team
from predict import DATA_DIR, Match, canonical_team, schedule


RESULTS_CSV = DATA_DIR / "world_cup_2026_results.csv"
MATCH_SHAPE_CSV = DATA_DIR / "match_shape_context.csv"
OUTPUT_CSV = DATA_DIR / "in_tournament_team_adjustments.csv"
EVENTS_CSV = DATA_DIR / "in_tournament_adjustment_events.csv"
POINTS_LIMIT = 180.0
ATTACK_LOW = 0.88
ATTACK_HIGH = 1.12
TEMPO_LOW = 0.90
TEMPO_HIGH = 1.14
RED_CARD_COLLAPSE_SIGNAL_WEIGHT = 0.50
RED_CARD_EVENT_SIGNAL_WEIGHT = 0.65
NO_STATE_UPDATE_LABELS = {"referee_lenient_anomaly", "officiating_anomaly"}
PREVIOUS_STATE_DECAY = 0.72
LATEST_SIGNAL_WEIGHT = 1.22
MANUAL_STATE_OVERRIDES = {
    "Canada": {
        "points_adjustment": 110.0,
        "attack_multiplier": 1.10,
        "tempo_multiplier": 1.14,
        "reason_suffix": "手动保留：Canada 本届进球能力此前已被多场复盘确认，2-1 Switzerland 不应把状态层重新压成负向",
    },
    "Cape Verde": {
        "points_adjustment": 180.0,
        "attack_multiplier": 1.12,
        "tempo_multiplier": 1.14,
        "reason_suffix": "手动校正：0-0 Spain 与 2-2 Uruguay 连续超预期，按本届状态层上限大幅上调强度和反击进球能力",
    },
    "Sweden": {
        "points_adjustment": 180.0,
        "attack_multiplier": 1.12,
        "tempo_multiplier": 1.08,
        "reason_suffix": "手动校正：5-1 Tunisia 与 1-1 Japan 说明本届进球能力和强度被低估",
    },
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
class AdjustmentSignal:
    result_surprise: float
    attack_surprise: float
    defense_surprise: float
    total_surprise: float
    points_signal: float
    attack_signal: float
    tempo_signal: float
    reason: str


@dataclass(frozen=True)
class MatchShape:
    observed_shapes: frozenset[str]


@dataclass
class TeamState:
    effective_after_bjt: datetime
    points_adjustment: float = 0.0
    attack_multiplier: float = 1.0
    tempo_multiplier: float = 1.0
    reason: str = ""


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def load_results() -> list[Result]:
    rows: list[Result] = []
    with RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                Result(
                    date_bjt=row["date_bjt"],
                    time_bjt=row["time_bjt"],
                    group=row["group"],
                    team_a=canonical_team(row["team_a"]),
                    team_b=canonical_team(row["team_b"]),
                    goals_a=int(row["goals_a"]),
                    goals_b=int(row["goals_b"]),
                )
            )
    return sorted(rows, key=result_datetime)


def load_match_shapes() -> dict[str, MatchShape]:
    if not MATCH_SHAPE_CSV.exists():
        return {}
    shapes: dict[str, MatchShape] = {}
    with MATCH_SHAPE_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            observed = frozenset(label for label in row["observed_shapes"].split(";") if label)
            shapes[row["match"]] = MatchShape(observed_shapes=observed)
    return shapes


def result_datetime(result: Result) -> datetime:
    return datetime.strptime(f"{result.date_bjt} {result.time_bjt}", "%Y-%m-%d %H:%M")


def result_key(result: Result) -> str:
    return "|".join([result.date_bjt, result.time_bjt, result.team_a, result.team_b])


def schedule_by_result_key() -> dict[tuple[str, str, str, str], Match]:
    matches: dict[tuple[str, str, str, str], Match] = {}
    for match in schedule():
        bjt = pfp.match_datetime_bjt(match)
        key = (
            bjt.strftime("%Y-%m-%d"),
            bjt.strftime("%H:%M"),
            canonical_team(match.team_a),
            canonical_team(match.team_b),
        )
        matches[key] = match
    return matches


def match_for_result(result: Result, matches: dict[tuple[str, str, str, str], Match]) -> Match:
    key = (result.date_bjt, result.time_bjt, result.team_a, result.team_b)
    if key in matches:
        return matches[key]
    reverse_key = (result.date_bjt, result.time_bjt, result.team_b, result.team_a)
    if reverse_key in matches:
        return matches[reverse_key]
    raise RuntimeError(f"result is missing from schedule: {result}")


def tournament_teams() -> set[str]:
    teams = {canonical_team(team) for match in schedule() for team in (match.team_a, match.team_b)}
    if len(teams) != 48:
        raise RuntimeError(f"expected 48 qualified teams, got {len(teams)}")
    return teams


def load_base_rankings() -> dict[str, pfp.FifaRanking]:
    teams = tournament_teams()
    rankings: dict[str, pfp.FifaRanking] = {}
    with pfp.FIFA_RANKING_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["year"]) != pfp.RANKING_YEAR:
                continue
            team = normalized_team(row["team"])
            if team not in teams:
                continue
            rankings[team] = pfp.FifaRanking(
                rank=int(row["rank"]),
                points=float(row["total_points"]),
                snapshot_date=row["snapshot_date"],
            )
    missing = sorted(teams - set(rankings))
    if missing:
        raise RuntimeError(f"missing base FIFA ranking for: {', '.join(missing)}")
    return rankings


def load_states() -> dict[str, TeamState]:
    states: dict[str, TeamState] = {}
    if not OUTPUT_CSV.exists():
        return states
    with OUTPUT_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            team = canonical_team(row["team"])
            states[team] = TeamState(
                effective_after_bjt=datetime.strptime(row["effective_after_bjt"], "%Y-%m-%d %H:%M"),
                points_adjustment=float(row["points_adjustment"]),
                attack_multiplier=float(row["attack_multiplier"]),
                tempo_multiplier=float(row["tempo_multiplier"]),
                reason=row["reason"],
            )
    return states


def active_adjustments(states: dict[str, TeamState]) -> dict[str, pfp.InTournamentAdjustment]:
    return {
        team: pfp.InTournamentAdjustment(
            effective_after_bjt=state.effective_after_bjt,
            points_adjustment=state.points_adjustment,
            attack_multiplier=state.attack_multiplier,
            tempo_multiplier=state.tempo_multiplier,
            reason=state.reason,
        )
        for team, state in states.items()
    }


def actual_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def calculate_signal(
    team: str,
    opponent: str,
    goals_for: int,
    goals_against: int,
    p_for: float,
    p_draw: float,
    xg_for: float,
    xg_against: float,
) -> AdjustmentSignal:
    expected_result = p_for + p_draw * 0.5
    result_surprise = actual_score(goals_for, goals_against) - expected_result
    attack_surprise = goals_for - xg_for
    defense_surprise = xg_against - goals_against
    total_surprise = goals_for + goals_against - xg_for - xg_against

    points = clamp(
        result_surprise * 150.0 + attack_surprise * 8.0 + defense_surprise * 6.0,
        -POINTS_LIMIT,
        POINTS_LIMIT,
    )
    attack = clamp(1.0 + attack_surprise * 0.025 + result_surprise * 0.025, ATTACK_LOW, ATTACK_HIGH)
    defense_leak = max(0.0, goals_against - xg_against)
    tempo = clamp(1.0 + total_surprise * 0.025 + defense_leak * 0.018, TEMPO_LOW, TEMPO_HIGH)
    reason = (
        f"自动状态层：{team} {goals_for}-{goals_against} {opponent}；"
        f"结果偏差{result_surprise:+.2f}，进攻偏差{attack_surprise:+.2f}球，"
        f"防守偏差{defense_surprise:+.2f}球"
    )
    return AdjustmentSignal(
        result_surprise=result_surprise,
        attack_surprise=attack_surprise,
        defense_surprise=defense_surprise,
        total_surprise=total_surprise,
        points_signal=points,
        attack_signal=attack,
        tempo_signal=tempo,
        reason=reason,
    )


def scale_signal(signal: AdjustmentSignal, weight: float, reason_suffix: str) -> AdjustmentSignal:
    if weight == 1.0:
        return signal
    attack = clamp(1.0 + (signal.attack_signal - 1.0) * weight, ATTACK_LOW, ATTACK_HIGH)
    tempo = clamp(1.0 + (signal.tempo_signal - 1.0) * weight, TEMPO_LOW, TEMPO_HIGH)
    return AdjustmentSignal(
        result_surprise=signal.result_surprise * weight,
        attack_surprise=signal.attack_surprise * weight,
        defense_surprise=signal.defense_surprise * weight,
        total_surprise=signal.total_surprise * weight,
        points_signal=signal.points_signal * weight,
        attack_signal=attack,
        tempo_signal=tempo,
        reason=f"{signal.reason}；{reason_suffix}",
    )


def update_team_state(
    states: dict[str, TeamState],
    team: str,
    effective_after_bjt: datetime,
    signal: AdjustmentSignal,
) -> TeamState:
    if team not in states:
        states[team] = TeamState(effective_after_bjt=effective_after_bjt)
    state = states[team]
    state.effective_after_bjt = effective_after_bjt
    weighted_points_signal = signal.points_signal * LATEST_SIGNAL_WEIGHT
    weighted_attack_signal = clamp(
        1.0 + (signal.attack_signal - 1.0) * LATEST_SIGNAL_WEIGHT,
        ATTACK_LOW,
        ATTACK_HIGH,
    )
    weighted_tempo_signal = clamp(
        1.0 + (signal.tempo_signal - 1.0) * LATEST_SIGNAL_WEIGHT,
        TEMPO_LOW,
        TEMPO_HIGH,
    )
    state.points_adjustment = clamp(
        state.points_adjustment * PREVIOUS_STATE_DECAY + weighted_points_signal,
        -POINTS_LIMIT,
        POINTS_LIMIT,
    )
    state.attack_multiplier = clamp(
        1.0 + (state.attack_multiplier - 1.0) * PREVIOUS_STATE_DECAY + (weighted_attack_signal - 1.0),
        ATTACK_LOW,
        ATTACK_HIGH,
    )
    state.tempo_multiplier = clamp(
        1.0 + (state.tempo_multiplier - 1.0) * PREVIOUS_STATE_DECAY + (weighted_tempo_signal - 1.0),
        TEMPO_LOW,
        TEMPO_HIGH,
    )
    state.reason = signal.reason
    if team in MANUAL_STATE_OVERRIDES:
        override = MANUAL_STATE_OVERRIDES[team]
        state.points_adjustment = max(state.points_adjustment, float(override["points_adjustment"]))
        state.attack_multiplier = max(state.attack_multiplier, float(override["attack_multiplier"]))
        state.tempo_multiplier = max(state.tempo_multiplier, float(override["tempo_multiplier"]))
        state.reason = f"{state.reason}；{override['reason_suffix']}"
    return state


def load_events() -> list[dict[str, str]]:
    if not EVENTS_CSV.exists():
        return []
    with EVENTS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def processed_results(events: list[dict[str, str]]) -> dict[str, tuple[int, int]]:
    processed: dict[str, tuple[int, int]] = {}
    for row in events:
        key = row["match_key"]
        score = (int(row["goals_a"]), int(row["goals_b"]))
        if key in processed and processed[key] != score:
            raise RuntimeError(f"event log has inconsistent score for {key}")
        processed[key] = score
    return processed


def event_row(
    result: Result,
    team: str,
    opponent: str,
    goals_for: int,
    goals_against: int,
    p_for: float,
    p_draw: float,
    xg_for: float,
    xg_against: float,
    signal: AdjustmentSignal,
    post_state: TeamState,
) -> dict[str, str]:
    return {
        "match_key": result_key(result),
        "date_bjt": result.date_bjt,
        "time_bjt": result.time_bjt,
        "group": result.group,
        "team_a": result.team_a,
        "team_b": result.team_b,
        "goals_a": str(result.goals_a),
        "goals_b": str(result.goals_b),
        "team": team,
        "opponent": opponent,
        "goals_for": str(goals_for),
        "goals_against": str(goals_against),
        "p_for": f"{p_for:.6f}",
        "p_draw": f"{p_draw:.6f}",
        "xg_for": f"{xg_for:.6f}",
        "xg_against": f"{xg_against:.6f}",
        "result_surprise": f"{signal.result_surprise:.6f}",
        "attack_surprise": f"{signal.attack_surprise:.6f}",
        "defense_surprise": f"{signal.defense_surprise:.6f}",
        "total_surprise": f"{signal.total_surprise:.6f}",
        "points_signal": f"{signal.points_signal:.6f}",
        "attack_signal": f"{signal.attack_signal:.6f}",
        "tempo_signal": f"{signal.tempo_signal:.6f}",
        "post_points_adjustment": f"{post_state.points_adjustment:.6f}",
        "post_attack_multiplier": f"{post_state.attack_multiplier:.6f}",
        "post_tempo_multiplier": f"{post_state.tempo_multiplier:.6f}",
        "reason": signal.reason,
    }


def apply_result(
    result: Result,
    states: dict[str, TeamState],
    matches: dict[tuple[str, str, str, str], Match],
    rankings: dict[str, pfp.FifaRanking],
    profiles: dict[str, pfp.TeamProfile],
    baselines: pfp.ProfileBaselines,
    market_values: dict[str, pfp.MarketValue],
    club_cohesion: dict[str, pfp.ClubCohesion],
    shapes: dict[str, MatchShape],
) -> list[dict[str, str]]:
    match = match_for_result(result, matches)
    pfp.IN_TOURNAMENT_ADJUSTMENTS = active_adjustments(states)
    prediction = pfp.predict_match(match, rankings, profiles, baselines, market_values, club_cohesion)
    if canonical_team(match.team_a) == result.team_a:
        goals_a, goals_b = result.goals_a, result.goals_b
    else:
        goals_a, goals_b = result.goals_b, result.goals_a

    match_time = result_datetime(result)
    p_a = float(prediction["p_a"])
    p_draw = float(prediction["p_draw"])
    p_b = float(prediction["p_b"])
    xg_a = float(prediction["xg_a"])
    xg_b = float(prediction["xg_b"])
    team_a = canonical_team(match.team_a)
    team_b = canonical_team(match.team_b)
    match_name = f"{team_a} vs {team_b}"
    shape = shapes.get(match_name)
    signal_weight = 1.0
    reason_suffix = ""
    if shape is not None:
        if shape.observed_shapes & NO_STATE_UPDATE_LABELS:
            signal_weight = 0.0
            reason_suffix = "裁判尺度/赛况异常，状态更新不用于调参"
        elif "red_card_collapse" in shape.observed_shapes:
            signal_weight = RED_CARD_COLLAPSE_SIGNAL_WEIGHT
            reason_suffix = "红牌崩盘赛况，状态更新降权50%"
        elif "red_card_event" in shape.observed_shapes:
            signal_weight = RED_CARD_EVENT_SIGNAL_WEIGHT
            reason_suffix = "红牌事件赛况，状态更新降权35%"

    signal_a = calculate_signal(team_a, team_b, goals_a, goals_b, p_a, p_draw, xg_a, xg_b)
    signal_a = scale_signal(signal_a, signal_weight, reason_suffix)
    post_a = update_team_state(states, team_a, match_time, signal_a)
    signal_b = calculate_signal(team_b, team_a, goals_b, goals_a, p_b, p_draw, xg_b, xg_a)
    signal_b = scale_signal(signal_b, signal_weight, reason_suffix)
    post_b = update_team_state(states, team_b, match_time, signal_b)
    return [
        event_row(result, team_a, team_b, goals_a, goals_b, p_a, p_draw, xg_a, xg_b, signal_a, post_a),
        event_row(result, team_b, team_a, goals_b, goals_a, p_b, p_draw, xg_b, xg_a, signal_b, post_b),
    ]


def build_rows_from_states(states: dict[str, TeamState]) -> list[dict[str, str]]:
    rows = []
    for team in sorted(states):
        state = states[team]
        rows.append(
            {
                "team": team,
                "effective_after_bjt": state.effective_after_bjt.strftime("%Y-%m-%d %H:%M"),
                "points_adjustment": f"{state.points_adjustment:.0f}",
                "attack_multiplier": f"{state.attack_multiplier:.2f}",
                "tempo_multiplier": f"{state.tempo_multiplier:.2f}",
                "reason": state.reason,
            }
        )
    return rows


def write_adjustments(rows: list[dict[str, str]]) -> None:
    fields = [
        "team",
        "effective_after_bjt",
        "points_adjustment",
        "attack_multiplier",
        "tempo_multiplier",
        "reason",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_events(rows: list[dict[str, str]]) -> None:
    fields = [
        "match_key",
        "date_bjt",
        "time_bjt",
        "group",
        "team_a",
        "team_b",
        "goals_a",
        "goals_b",
        "team",
        "opponent",
        "goals_for",
        "goals_against",
        "p_for",
        "p_draw",
        "xg_for",
        "xg_against",
        "result_surprise",
        "attack_surprise",
        "defense_surprise",
        "total_surprise",
        "points_signal",
        "attack_signal",
        "tempo_signal",
        "post_points_adjustment",
        "post_attack_multiplier",
        "post_tempo_multiplier",
        "reason",
    ]
    with EVENTS_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_update(*, replay: bool) -> tuple[int, int]:
    rankings = load_base_rankings()
    profiles = pfp.load_profiles()
    market_values = pfp.load_market_values()
    club_cohesion = pfp.load_club_cohesion()
    baselines = pfp.profile_baselines(list(profiles.values()))
    matches = schedule_by_result_key()
    shapes = load_match_shapes()
    results = load_results()
    events = [] if replay or not EVENTS_CSV.exists() else load_events()
    processed = processed_results(events)
    states = {} if replay or not EVENTS_CSV.exists() else load_states()
    new_events: list[dict[str, str]] = []

    for result in results:
        key = result_key(result)
        if key in processed:
            score = processed[key]
            if score != (result.goals_a, result.goals_b):
                raise RuntimeError(
                    f"processed result changed for {key}: events have {score[0]}-{score[1]}, "
                    f"results have {result.goals_a}-{result.goals_b}; run with --replay after reviewing the correction"
                )
            continue
        new_events.extend(
            apply_result(
                result,
                states,
                matches,
                rankings,
                profiles,
                baselines,
                market_values,
                club_cohesion,
                shapes,
            )
        )

    if replay:
        events = new_events
    else:
        events.extend(new_events)
    write_events(events)
    write_adjustments(build_rows_from_states(states))
    return len(new_events) // 2, len(events) // 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update in-tournament team state from newly completed matches.")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Rebuild the event ledger and state snapshot from the results CSV. Use only for migration or corrected old scores.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = parse_args()
    added_matches, total_matches = run_update(replay=args.replay)
    mode = "replay" if args.replay or not EVENTS_CSV.exists() else "incremental"
    print(f"Mode: {mode}")
    print(f"CSV: {OUTPUT_CSV}")
    print(f"Events: {EVENTS_CSV}")
    print(f"New matches processed: {added_matches}")
    print(f"Total processed matches: {total_matches}")


if __name__ == "__main__":
    main()
