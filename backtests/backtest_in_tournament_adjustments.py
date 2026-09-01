from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from backtests import backtest_world_cup_fifa_profile_scores as base
from backtests.backtest_world_cup_fifa_ranking import WORLD_CUPS, WorldCupMatch, load_world_cup_matches


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
MATCH_CSV = OUTPUT_DIR / "in_tournament_adjustment_backtest_matches.csv"
SUMMARY_MD = OUTPUT_DIR / "in_tournament_adjustment_backtest_summary.md"
POINTS_LIMIT = 180.0
ATTACK_LOW = 0.88
ATTACK_HIGH = 1.12
TEMPO_LOW = 0.90
TEMPO_HIGH = 1.14
POINT_SIGNAL_RESULT_WEIGHT = 150.0
POINT_SIGNAL_ATTACK_WEIGHT = 8.0
POINT_SIGNAL_DEFENSE_WEIGHT = 6.0
ATTACK_SIGNAL_ATTACK_WEIGHT = 0.025
ATTACK_SIGNAL_RESULT_WEIGHT = 0.025
TEMPO_SIGNAL_TOTAL_WEIGHT = 0.025
TEMPO_SIGNAL_DEFENSE_LEAK_WEIGHT = 0.018
EDGE_COMPRESSION_WEIGHT = 0.08
EDGE_COMPRESSION_LIMIT = 0.22
EDGE_SHIFT_POINTS = 10.0
DRAW_BOTH_UNDER_WEIGHT = 0.035
DRAW_DIFF_WEIGHT = 0.035
TEMPO_STATE_WEIGHT = 0.55
TEMPO_UNDER_WEIGHT = 0.09
SPLIT_SIGNAL_WEIGHT = 0.03
SPLIT_ATTACK_WEIGHT = 0.02
GOAL_SUPPRESSION_STRONG = 0.48
GOAL_SUPPRESSION_MEDIUM = 0.28
LOW_BUCKET_SUPPRESSION = 0.92
LOW_BUCKET_DRAW = 0.37


@dataclass
class TeamState:
    points_adjustment: float = 0.0
    attack_multiplier: float = 1.0
    tempo_multiplier: float = 1.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def actual_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def parse_score(value: str) -> tuple[int, int]:
    home, away = value.split("-", maxsplit=1)
    return int(home), int(away)


def score_distance(actual: str, predicted: str) -> int:
    actual_home, actual_away = parse_score(actual)
    predicted_home, predicted_away = parse_score(predicted)
    return abs(actual_home - predicted_home) + abs(actual_away - predicted_away)


def score_deviation(actual: str, predicted: str) -> float:
    actual_home, actual_away = parse_score(actual)
    denominator = max(1, actual_home + actual_away)
    return score_distance(actual, predicted) / denominator


def score_mean_deviation(actual: str, predictions: list[str]) -> float:
    if not predictions:
        raise ValueError("predictions cannot be empty")
    return sum(score_deviation(actual, prediction) for prediction in predictions) / len(predictions)


def team_adjusted_points(points: float, states: dict[str, TeamState], team: str) -> float:
    if team not in states:
        return points
    return points + states[team].points_adjustment


def state_signal(states: dict[str, TeamState], team: str) -> float:
    if team not in states:
        return 0.0
    return clamp(states[team].points_adjustment / POINTS_LIMIT, -1.0, 1.0)


def adjusted_point_edge(raw_point_edge: float, states: dict[str, TeamState], home_team: str, away_team: str) -> float:
    signal_home = state_signal(states, home_team)
    signal_away = state_signal(states, away_team)
    if signal_home == 0.0 and signal_away == 0.0:
        return raw_point_edge
    if raw_point_edge == 0.0:
        return (signal_home - signal_away) * EDGE_SHIFT_POINTS

    favorite_signal = signal_home if raw_point_edge > 0 else signal_away
    underdog_signal = signal_away if raw_point_edge > 0 else signal_home
    compression = clamp((underdog_signal - favorite_signal) * EDGE_COMPRESSION_WEIGHT, -EDGE_COMPRESSION_LIMIT, EDGE_COMPRESSION_LIMIT)
    adjusted_abs_edge = abs(raw_point_edge) * (1.0 - compression)
    adjusted_abs_edge = clamp(adjusted_abs_edge, 0.0, abs(raw_point_edge) * (1.0 + EDGE_COMPRESSION_LIMIT))
    adjusted = base.math.copysign(adjusted_abs_edge, raw_point_edge)
    adjusted += (signal_home - signal_away) * EDGE_SHIFT_POINTS
    return adjusted


def adjusted_draw_probability(base_draw: float, states: dict[str, TeamState], home_team: str, away_team: str) -> float:
    signal_home = state_signal(states, home_team)
    signal_away = state_signal(states, away_team)
    if signal_home == 0.0 and signal_away == 0.0:
        return base_draw
    convergence = max(0.0, 1.0 - abs(signal_home - signal_away))
    both_under = max(0.0, -signal_home) + max(0.0, -signal_away)
    one_over_one_under = max(0.0, max(signal_home, signal_away) - min(signal_home, signal_away))
    draw_bonus = convergence * both_under * DRAW_BOTH_UNDER_WEIGHT + min(one_over_one_under, 1.4) * DRAW_DIFF_WEIGHT
    if state_goal_suppression(states, home_team, away_team) >= GOAL_SUPPRESSION_STRONG:
        draw_bonus += 0.025
    return clamp(base_draw + draw_bonus, 0.16, 0.42)


def state_goal_suppression(states: dict[str, TeamState], home_team: str, away_team: str) -> float:
    if home_team not in states and away_team not in states:
        return 0.0
    signal_home = state_signal(states, home_team)
    signal_away = state_signal(states, away_team)
    compression = abs(signal_home - signal_away) * 0.35
    both_under = (max(0.0, -signal_home) + max(0.0, -signal_away)) * 0.55
    one_under = max(0.0, -min(signal_home, signal_away)) * 0.45
    return clamp(compression + both_under + one_under, 0.0, 1.0)


def lower_total_goal_bucket(bucket: str) -> str:
    if bucket == "6-8球":
        return "4-5球"
    if bucket == "4-5球":
        return "2-3球"
    if bucket == "2-3球":
        return "0-1球"
    return bucket


def apply_state_total_goal_suppression(
    selected_bucket: str,
    buckets: list[tuple[str, float]],
    states: dict[str, TeamState],
    home_team: str,
    away_team: str,
    p_draw: float,
) -> str:
    suppression = state_goal_suppression(states, home_team, away_team)
    if suppression < GOAL_SUPPRESSION_MEDIUM:
        return selected_bucket
    probabilities = dict(buckets)
    lower_bucket = lower_total_goal_bucket(selected_bucket)
    if lower_bucket == selected_bucket:
        return selected_bucket
    if selected_bucket == "2-3球":
        if suppression < LOW_BUCKET_SUPPRESSION or p_draw < LOW_BUCKET_DRAW:
            return selected_bucket
    selected_probability = probabilities.get(selected_bucket, 0.0)
    lower_probability = probabilities.get(lower_bucket, 0.0)
    threshold = 0.42 if suppression >= GOAL_SUPPRESSION_STRONG else 0.72
    if lower_probability >= selected_probability * threshold:
        return lower_bucket
    return selected_bucket


def choose_suppressed_second_total_goal_bucket(
    selected_bucket: str,
    buckets: list[tuple[str, float]],
    states: dict[str, TeamState],
    home_team: str,
    away_team: str,
) -> str | None:
    if state_goal_suppression(states, home_team, away_team) < GOAL_SUPPRESSION_MEDIUM:
        return None
    lower_bucket = lower_total_goal_bucket(selected_bucket)
    if lower_bucket != selected_bucket:
        return lower_bucket
    return next((bucket for bucket, _ in buckets if bucket != selected_bucket), None)


def state_tempo_multiplier(states: dict[str, TeamState], home_team: str, away_team: str) -> float:
    multiplier = 1.0
    active = False
    for team in (home_team, away_team):
        if team in states:
            signal = state_signal(states, team)
            state = states[team]
            multiplier *= 1.0 + (state.tempo_multiplier - 1.0) * TEMPO_STATE_WEIGHT
            multiplier *= 1.0 - max(0.0, -signal) * TEMPO_UNDER_WEIGHT
            active = True
    if not active:
        return 1.0
    return clamp(multiplier, 0.82, 1.18)


def apply_attack_split_adjustment(
    lambda_home: float,
    lambda_away: float,
    states: dict[str, TeamState],
    home_team: str,
    away_team: str,
) -> tuple[float, float]:
    if home_team not in states and away_team not in states:
        return lambda_home, lambda_away
    total = lambda_home + lambda_away
    if total <= 0:
        return lambda_home, lambda_away
    signal_home = state_signal(states, home_team)
    signal_away = state_signal(states, away_team)
    split_home = lambda_home / total
    split_home += (signal_home - signal_away) * SPLIT_SIGNAL_WEIGHT
    if home_team in states:
        split_home += (states[home_team].attack_multiplier - 1.0) * SPLIT_ATTACK_WEIGHT
    if away_team in states:
        split_home -= (states[away_team].attack_multiplier - 1.0) * SPLIT_ATTACK_WEIGHT
    split_home = clamp(split_home, 0.18, 0.82)
    return total * split_home, total * (1.0 - split_home)


def outcome_probabilities(
    match: WorldCupMatch,
    model: base.YearModel,
    states: dict[str, TeamState],
) -> tuple[float, float, float]:
    home_rank = model.rankings[match.home_team]
    away_rank = model.rankings[match.away_team]
    home_profile = model.profiles[match.home_team]
    away_profile = model.profiles[match.away_team]
    point_edge = adjusted_point_edge(home_rank.points - away_rank.points, states, match.home_team, match.away_team)
    point_edge += base.host_edge(match.home_team, match)
    point_edge -= base.host_edge(match.away_team, match)
    non_draw_home = 1.0 / (1.0 + base.math.exp(-point_edge / base.POINT_EDGE_SCALE))
    p_draw = adjusted_draw_probability(
        base.draw_probability(abs(home_rank.rank - away_rank.rank), home_profile, away_profile),
        states,
        match.home_team,
        match.away_team,
    )
    non_draw_mass = 1.0 - p_draw
    return non_draw_mass * non_draw_home, p_draw, non_draw_mass * (1.0 - non_draw_home)


def expected_goals(
    match: WorldCupMatch,
    model: base.YearModel,
    base_goals_per_match_by_stage: dict[str, float],
    states: dict[str, TeamState],
) -> tuple[float, float]:
    home_rank = model.rankings[match.home_team]
    away_rank = model.rankings[match.away_team]
    home_profile = model.profiles[match.home_team]
    away_profile = model.profiles[match.away_team]
    p_home, _, p_away = outcome_probabilities(match, model, states)

    base_goals = base_goals_per_match_by_stage[base.stage_bucket(match)]
    total_goals = base_goals * base.style_total_modifier(home_profile, away_profile, model.baselines)
    total_goals *= state_tempo_multiplier(states, match.home_team, match.away_team)
    attack_home = home_profile.goals_for / max(0.2, home_profile.goals_for + away_profile.goals_for)
    defense_away = away_profile.goals_against / max(0.2, home_profile.goals_against + away_profile.goals_against)
    split_home = 0.50 + (p_home - p_away) * 0.55
    split_home += (attack_home - 0.5) * 0.30
    split_home += (defense_away - 0.5) * 0.22
    split_home += (home_profile.multi_goal_rate - away_profile.conceded_multi_rate) * 0.08
    split_home -= (away_profile.clean_sheet_rate - home_profile.clean_sheet_rate) * 0.06
    split_home += (away_rank.rank - home_rank.rank) / 400.0
    split_home += base.HOST_SPLIT_ADJUSTMENT if base.host_edge(match.home_team, match) else 0.0
    split_home -= base.HOST_SPLIT_ADJUSTMENT if base.host_edge(match.away_team, match) else 0.0

    split_home = base.clamp(split_home, 0.18, 0.82)
    lambda_home = base.clamp(total_goals * split_home, 0.05, 4.2)
    lambda_away = base.clamp(total_goals * (1.0 - split_home), 0.05, 4.2)
    lambda_home, lambda_away = apply_attack_split_adjustment(
        lambda_home,
        lambda_away,
        states,
        match.home_team,
        match.away_team,
    )
    return base.clamp(lambda_home, 0.05, 4.2), base.clamp(lambda_away, 0.05, 4.2)


def predict_with_states(
    match: WorldCupMatch,
    model: base.YearModel,
    base_goals_per_match_by_stage: dict[str, float],
    states: dict[str, TeamState],
) -> dict:
    p_home, p_draw, p_away = outcome_probabilities(match, model, states)
    lambda_home, lambda_away = expected_goals(match, model, base_goals_per_match_by_stage, states)
    cells = base.outcome_adjusted_scores(lambda_home, lambda_away, p_home, p_draw, p_away)
    predicted_outcome = base.predicted_outcome_from_probabilities(
        p_home,
        p_draw,
        p_away,
        home_label="home",
        draw_label="draw",
        away_label="away",
    )
    recommended, _, total_goals = base.select_recommended_score(cells, predicted_outcome)
    buckets = base.top_total_goal_buckets(total_goals)
    selected_bucket = apply_state_total_goal_suppression(
        buckets[0][0],
        buckets,
        states,
        match.home_team,
        match.away_team,
        p_draw,
    )
    complement_bucket = choose_suppressed_second_total_goal_bucket(
        selected_bucket,
        buckets,
        states,
        match.home_team,
        match.away_team,
    ) or next(bucket for bucket, _ in buckets if bucket != selected_bucket)
    bucket_primary_score = base.best_score_inside_total_goal_buckets(cells, {selected_bucket})
    bucket_complement_score = base.best_score_inside_total_goal_buckets(cells, {complement_bucket})
    actual = base.outcome(match.home_score, match.away_score)
    actual_total = match.home_score + match.away_score
    actual_bucket = base.top_total_goal_buckets([(actual_total, 1.0)])[0][0]
    return {
        "predicted_outcome": predicted_outcome,
        "outcome_correct": predicted_outcome == actual,
        "recommended_score": f"{recommended[0]}-{recommended[1]}",
        "score_correct": recommended[0] == match.home_score and recommended[1] == match.away_score,
        "bucket_primary_score": f"{bucket_primary_score[0]}-{bucket_primary_score[1]}",
        "bucket_complement_score": f"{bucket_complement_score[0]}-{bucket_complement_score[1]}",
        "selected_total_goal_bucket": selected_bucket,
        "top2_total_goal_bucket": complement_bucket,
        "top1_total_goal_bucket_hit": actual_bucket == selected_bucket,
        "top2_total_goal_bucket_hit": actual_bucket in {selected_bucket, complement_bucket},
        "xg_home": lambda_home,
        "xg_away": lambda_away,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
    }


def calculate_signal(
    goals_for: int,
    goals_against: int,
    p_for: float,
    p_draw: float,
    xg_for: float,
    xg_against: float,
) -> tuple[float, float, float]:
    expected_result = p_for + p_draw * 0.5
    result_surprise = actual_score(goals_for, goals_against) - expected_result
    attack_surprise = goals_for - xg_for
    defense_surprise = xg_against - goals_against
    total_surprise = goals_for + goals_against - xg_for - xg_against
    points = clamp(
        result_surprise * POINT_SIGNAL_RESULT_WEIGHT
        + attack_surprise * POINT_SIGNAL_ATTACK_WEIGHT
        + defense_surprise * POINT_SIGNAL_DEFENSE_WEIGHT,
        -POINTS_LIMIT,
        POINTS_LIMIT,
    )
    attack = clamp(
        1.0 + attack_surprise * ATTACK_SIGNAL_ATTACK_WEIGHT + result_surprise * ATTACK_SIGNAL_RESULT_WEIGHT,
        ATTACK_LOW,
        ATTACK_HIGH,
    )
    defense_leak = max(0.0, goals_against - xg_against)
    tempo = clamp(
        1.0 + total_surprise * TEMPO_SIGNAL_TOTAL_WEIGHT + defense_leak * TEMPO_SIGNAL_DEFENSE_LEAK_WEIGHT,
        TEMPO_LOW,
        TEMPO_HIGH,
    )
    return points, attack, tempo


def update_state(states: dict[str, TeamState], team: str, points: float, attack: float, tempo: float) -> None:
    if team not in states:
        states[team] = TeamState()
    state = states[team]
    state.points_adjustment = clamp(state.points_adjustment + points, -POINTS_LIMIT, POINTS_LIMIT)
    state.attack_multiplier = clamp(state.attack_multiplier * attack, ATTACK_LOW, ATTACK_HIGH)
    state.tempo_multiplier = clamp(state.tempo_multiplier * tempo, TEMPO_LOW, TEMPO_HIGH)


def update_states_from_match(
    states: dict[str, TeamState],
    match: WorldCupMatch,
    prediction: dict,
) -> None:
    home_signal = calculate_signal(
        match.home_score,
        match.away_score,
        float(prediction["p_home"]),
        float(prediction["p_draw"]),
        float(prediction["xg_home"]),
        float(prediction["xg_away"]),
    )
    away_signal = calculate_signal(
        match.away_score,
        match.home_score,
        float(prediction["p_away"]),
        float(prediction["p_draw"]),
        float(prediction["xg_away"]),
        float(prediction["xg_home"]),
    )
    update_state(states, match.home_team, *home_signal)
    update_state(states, match.away_team, *away_signal)


def accuracy(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row[key]) / len(rows)


def build_rows(matches: list[WorldCupMatch], models: dict[int, base.YearModel]) -> list[dict]:
    rows: list[dict] = []
    all_years = set(WORLD_CUPS)
    for target_year in sorted(WORLD_CUPS):
        training_years = all_years - {target_year}
        base_goals = base.goals_per_match_by_stage(matches, training_years)
        states: dict[str, TeamState] = {}
        for match in [item for item in matches if item.year == target_year]:
            model = models[target_year]
            home_has_prior_state = match.home_team in states
            away_has_prior_state = match.away_team in states
            baseline = base.predict_match(match, model, base_goals)
            adjusted = predict_with_states(match, model, base_goals, states)
            actual_bucket = baseline["actual_total_goal_bucket"]
            actual_score = f"{match.home_score}-{match.away_score}"
            baseline_candidates = [
                baseline["recommended_score"],
                baseline["bucket_primary_score"],
                baseline["bucket_complement_score"],
            ]
            adjusted_candidates = [
                adjusted["recommended_score"],
                adjusted["bucket_primary_score"],
                adjusted["bucket_complement_score"],
            ]
            row = {
                "year": target_year,
                "round_index": match.round_index,
                "stage": base.stage_bucket(match),
                "date": match.date.isoformat(),
                "home_team": match.home_team,
                "away_team": match.away_team,
                "actual_score": actual_score,
                "actual_total_goal_bucket": actual_bucket,
                "baseline_outcome": baseline["predicted_outcome"],
                "adjusted_outcome": adjusted["predicted_outcome"],
                "baseline_outcome_correct": baseline["outcome_correct"],
                "adjusted_outcome_correct": adjusted["outcome_correct"],
                "baseline_score": baseline["recommended_score"],
                "adjusted_score": adjusted["recommended_score"],
                "baseline_score_candidates": ";".join(baseline_candidates),
                "adjusted_score_candidates": ";".join(adjusted_candidates),
                "baseline_score_distance": score_distance(
                    actual_score,
                    baseline["recommended_score"],
                ),
                "adjusted_score_distance": score_distance(
                    actual_score,
                    adjusted["recommended_score"],
                ),
                "baseline_score_deviation": score_deviation(
                    actual_score,
                    baseline["recommended_score"],
                ),
                "adjusted_score_deviation": score_deviation(
                    actual_score,
                    adjusted["recommended_score"],
                ),
                "baseline_score_mean_deviation": score_mean_deviation(actual_score, baseline_candidates),
                "adjusted_score_mean_deviation": score_mean_deviation(actual_score, adjusted_candidates),
                "baseline_score_correct": baseline["score_correct"],
                "adjusted_score_correct": adjusted["score_correct"],
                "baseline_total_bucket": baseline["selected_total_goal_bucket"],
                "adjusted_total_bucket": adjusted["selected_total_goal_bucket"],
                "baseline_top1_bucket_hit": baseline["top1_total_goal_bucket_hit"],
                "adjusted_top1_bucket_hit": adjusted["top1_total_goal_bucket_hit"],
                "baseline_top2_bucket_hit": baseline["top2_total_goal_bucket_hit"],
                "adjusted_top2_bucket_hit": adjusted["top2_total_goal_bucket_hit"],
                "baseline_valid_for_score_deviation": (
                    baseline["outcome_correct"] and baseline["top1_total_goal_bucket_hit"]
                ),
                "adjusted_valid_for_score_deviation": (
                    adjusted["outcome_correct"] and adjusted["top1_total_goal_bucket_hit"]
                ),
                "common_valid_for_score_deviation": (
                    baseline["outcome_correct"]
                    and adjusted["outcome_correct"]
                    and baseline["top1_total_goal_bucket_hit"]
                    and adjusted["top1_total_goal_bucket_hit"]
                ),
                "has_prior_state": home_has_prior_state or away_has_prior_state,
                "both_have_prior_state": home_has_prior_state and away_has_prior_state,
                "home_prior_points_adjustment": states.get(match.home_team, TeamState()).points_adjustment,
                "away_prior_points_adjustment": states.get(match.away_team, TeamState()).points_adjustment,
            }
            rows.append(row)
            update_states_from_match(states, match, adjusted)
    return rows


def write_match_csv(rows: list[dict]) -> None:
    fields = [
        "year",
        "round_index",
        "stage",
        "date",
        "home_team",
        "away_team",
        "actual_score",
        "actual_total_goal_bucket",
        "baseline_outcome",
        "adjusted_outcome",
        "baseline_outcome_correct",
        "adjusted_outcome_correct",
        "baseline_score",
        "adjusted_score",
        "baseline_score_candidates",
        "adjusted_score_candidates",
        "baseline_score_distance",
        "adjusted_score_distance",
        "baseline_score_deviation",
        "adjusted_score_deviation",
        "baseline_score_mean_deviation",
        "adjusted_score_mean_deviation",
        "baseline_score_correct",
        "adjusted_score_correct",
        "baseline_total_bucket",
        "adjusted_total_bucket",
        "baseline_top1_bucket_hit",
        "adjusted_top1_bucket_hit",
        "baseline_top2_bucket_hit",
        "adjusted_top2_bucket_hit",
        "baseline_valid_for_score_deviation",
        "adjusted_valid_for_score_deviation",
        "common_valid_for_score_deviation",
        "has_prior_state",
        "both_have_prior_state",
        "home_prior_points_adjustment",
        "away_prior_points_adjustment",
    ]
    with MATCH_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_line(label: str, rows: list[dict]) -> str:
    return (
        f"| {label} | {len(rows)} | "
        f"{base.pct(accuracy(rows, 'baseline_outcome_correct'))} | {base.pct(accuracy(rows, 'adjusted_outcome_correct'))} | "
        f"{base.pct(accuracy(rows, 'baseline_top1_bucket_hit'))} | {base.pct(accuracy(rows, 'adjusted_top1_bucket_hit'))} | "
        f"{base.pct(accuracy(rows, 'baseline_top2_bucket_hit'))} | {base.pct(accuracy(rows, 'adjusted_top2_bucket_hit'))} | "
        f"{base.pct(accuracy(rows, 'baseline_score_correct'))} | {base.pct(accuracy(rows, 'adjusted_score_correct'))} |"
    )


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def trimmed_mean(values: list[float], trim_ratio: float = 0.10) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    trim = int(len(ordered) * trim_ratio)
    trimmed = ordered[trim:len(ordered) - trim] if trim else ordered
    if not trimmed:
        trimmed = ordered
    return sum(trimmed) / len(trimmed)


def score_deviation_values(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def deviation_summary_line(label: str, rows: list[dict]) -> str:
    baseline = score_deviation_values(rows, "baseline_score_mean_deviation")
    adjusted = score_deviation_values(rows, "adjusted_score_mean_deviation")
    baseline_median = median(baseline)
    adjusted_median = median(adjusted)
    baseline_trimmed = trimmed_mean(baseline)
    adjusted_trimmed = trimmed_mean(adjusted)
    return (
        f"| {label} | {len(rows)} | "
        f"{baseline_median:.3f} | {adjusted_median:.3f} | {baseline_median - adjusted_median:+.3f} | "
        f"{baseline_trimmed:.3f} | {adjusted_trimmed:.3f} | {baseline_trimmed - adjusted_trimmed:+.3f} |"
    )


def valid_deviation_summary_line(label: str, rows: list[dict]) -> str:
    baseline_rows = [row for row in rows if row["baseline_valid_for_score_deviation"]]
    adjusted_rows = [row for row in rows if row["adjusted_valid_for_score_deviation"]]
    common_rows = [row for row in rows if row["common_valid_for_score_deviation"]]
    baseline_valid = score_deviation_values(baseline_rows, "baseline_score_mean_deviation")
    adjusted_valid = score_deviation_values(adjusted_rows, "adjusted_score_mean_deviation")
    baseline_common = score_deviation_values(common_rows, "baseline_score_mean_deviation")
    adjusted_common = score_deviation_values(common_rows, "adjusted_score_mean_deviation")
    return (
        f"| {label} | {len(baseline_rows)} | {len(adjusted_rows)} | {len(common_rows)} | "
        f"{median(baseline_valid):.3f} | {median(adjusted_valid):.3f} | "
        f"{trimmed_mean(baseline_common):.3f} | {trimmed_mean(adjusted_common):.3f} | "
        f"{trimmed_mean(baseline_common) - trimmed_mean(adjusted_common):+.3f} |"
    )


def write_summary(rows: list[dict]) -> None:
    prior_state_rows = [row for row in rows if row["has_prior_state"]]
    both_prior_state_rows = [row for row in rows if row["both_have_prior_state"]]
    group_prior_state_rows = [row for row in prior_state_rows if row["stage"] == "group"]
    knockout_rows = [row for row in rows if row["stage"] == "knockout"]
    lines = [
        "# 赛中状态层历史验证",
        "",
        "- 预测每场前只使用该年份此前已经完成的比赛生成状态层。",
        "- 当前场赛果只在预测后写入状态，用于后续场次。",
        "- 没有赛前状态的球队不会被修正；重点看 `有赛前状态` 的比赛。",
        "- 核心指标是三比分平均偏离度，越低越好。单个比分偏离度：`(|预测主队进球-实际主队进球| + |预测客队进球-实际客队进球|) / max(1, 实际总进球)`。",
        "- 三比分平均偏离度取模型候选、Top1 桶候选、Top2 桶候选三个比分的均值。",
        "- 汇总同时看中位数和 10% 截尾均值，避免极端比分主导判断。",
        "",
        "## 比分偏离度",
        "",
        "| 范围 | 场次 | 基线中位数 | 状态层中位数 | 中位数改善 | 基线截尾均值 | 状态层截尾均值 | 截尾均值改善 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        deviation_summary_line("全部", rows),
        deviation_summary_line("有赛前状态", prior_state_rows),
        deviation_summary_line("小组赛且有赛前状态", group_prior_state_rows),
        deviation_summary_line("淘汰赛", knockout_rows),
        "",
        "## 胜负和Top1总进球桶都命中时",
        "",
        "| 范围 | 基线可算场次 | 状态层可算场次 | 共同可算场次 | 基线中位数 | 状态层中位数 | 共同基线截尾均值 | 共同状态层截尾均值 | 共同改善 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        valid_deviation_summary_line("全部", rows),
        valid_deviation_summary_line("有赛前状态", prior_state_rows),
        valid_deviation_summary_line("小组赛且有赛前状态", group_prior_state_rows),
        valid_deviation_summary_line("淘汰赛", knockout_rows),
        "",
        "## 命中率辅助指标",
        "",
        "| 范围 | 场次 | 基线赛果 | 状态层赛果 | 基线Top1桶 | 状态层Top1桶 | 基线Top2桶 | 状态层Top2桶 | 基线比分 | 状态层比分 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        summary_line("全部", rows),
        summary_line("有赛前状态", prior_state_rows),
        summary_line("双方有赛前状态", both_prior_state_rows),
        summary_line("小组赛且有赛前状态", group_prior_state_rows),
        summary_line("淘汰赛", knockout_rows),
    ]
    for year in sorted(WORLD_CUPS):
        year_rows = [row for row in rows if row["year"] == year]
        prior_rows = [row for row in year_rows if row["has_prior_state"]]
        lines.append(summary_line(str(year), year_rows))
        lines.append(summary_line(f"{year}有赛前状态", prior_rows))
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    matches = load_world_cup_matches()
    models = base.build_year_models(matches)
    rows = build_rows(matches, models)
    write_match_csv(rows)
    write_summary(rows)
    prior_state_rows = [row for row in rows if row["has_prior_state"]]
    prior_baseline = score_deviation_values(prior_state_rows, "baseline_score_mean_deviation")
    prior_adjusted = score_deviation_values(prior_state_rows, "adjusted_score_mean_deviation")
    print(f"Matches: {MATCH_CSV}")
    print(f"Summary: {SUMMARY_MD}")
    print(
        "Prior-state three-score mean deviation median: "
        f"{median(prior_baseline):.3f} -> {median(prior_adjusted):.3f}"
    )
    print(
        "Prior-state three-score mean deviation trimmed mean: "
        f"{trimmed_mean(prior_baseline):.3f} -> {trimmed_mean(prior_adjusted):.3f}"
    )
    print(
        "Prior-state Top1 bucket: "
        f"{base.pct(accuracy(prior_state_rows, 'baseline_top1_bucket_hit'))} -> "
        f"{base.pct(accuracy(prior_state_rows, 'adjusted_top1_bucket_hit'))}"
    )
    print(
        "Prior-state outcome: "
        f"{base.pct(accuracy(prior_state_rows, 'baseline_outcome_correct'))} -> "
        f"{base.pct(accuracy(prior_state_rows, 'adjusted_outcome_correct'))}"
    )


if __name__ == "__main__":
    main()
