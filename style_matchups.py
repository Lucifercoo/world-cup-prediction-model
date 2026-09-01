from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from predict import DATA_DIR, OUTPUT_DIR


STYLE_MATCHUP_EDGES_CSV = DATA_DIR / "style_matchup_edges.csv"

MIN_EDGE_WEIGHT = 35.0
MIN_ABS_EDGE = 0.010
MAX_PAIR_FEATURES = 6
STYLE_CLOSE_FULL_POINT_EDGE = 80.0
STYLE_CLOSE_ZERO_POINT_EDGE = 260.0
STYLE_CLOSE_FULL_PROB_EDGE = 0.08
STYLE_CLOSE_ZERO_PROB_EDGE = 0.32
STYLE_MIN_INFLUENCE_FACTOR = 0.15

STYLE_FEATURES = {
    "攻守兼备型": "style_complete",
    "进攻型": "style_attacking",
    "防守型": "style_defensive",
    "开放型": "style_open",
    "低效型": "style_low_efficiency",
    "均衡型": "style_balanced",
}

TEAM_SHAPE_FEATURES = {
    "low_event_team": "low_event",
    "strong_defense_attack_suppression_team": "suppression",
    "control_team": "control",
    "transition_route_team": "transition_route",
    "open_event_team": "high_event",
    "defensive_resistance_team": "defense_resistance",
    "defensive_fragility_team": "defense_fragility",
}


@dataclass(frozen=True)
class StyleEdge:
    feature_a: str
    feature_b: str
    weighted_matches: float
    residual_points_share_a: float
    shrunk_residual: float
    total_goal_multiplier: float


@dataclass(frozen=True)
class StyleMatchupEffect:
    edge: float
    total_goal_multiplier: float
    reasons: tuple[str, ...]

    @property
    def points_shift(self) -> float:
        return self.edge * 240.0

    @property
    def xg_split_shift(self) -> float:
        return self.edge * 0.24

    @property
    def probability_scale_a(self) -> float:
        return max(0.90, min(1.10, 1.0 + self.edge * 0.70))

    @property
    def probability_scale_b(self) -> float:
        return max(0.90, min(1.10, 1.0 - self.edge * 0.70))

    @property
    def xg_scale_a(self) -> float:
        return max(0.92, min(1.08, 1.0 + self.edge * 0.75))

    @property
    def xg_scale_b(self) -> float:
        return max(0.92, min(1.08, 1.0 - self.edge * 0.75))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _linear_closeness(value: float, full_at: float, zero_at: float) -> float:
    if value <= full_at:
        return 1.0
    if value >= zero_at:
        return 0.0
    return 1.0 - (value - full_at) / (zero_at - full_at)


def style_influence_factor(
    *,
    point_edge: float | None = None,
    p_a: float | None = None,
    p_b: float | None = None,
) -> float:
    closeness_values: list[float] = []
    if point_edge is not None:
        closeness_values.append(
            _linear_closeness(abs(point_edge), STYLE_CLOSE_FULL_POINT_EDGE, STYLE_CLOSE_ZERO_POINT_EDGE)
        )
    if p_a is not None and p_b is not None:
        closeness_values.append(
            _linear_closeness(abs(p_a - p_b), STYLE_CLOSE_FULL_PROB_EDGE, STYLE_CLOSE_ZERO_PROB_EDGE)
        )
    if not closeness_values:
        return 1.0
    closeness = min(closeness_values)
    return clamp(
        STYLE_MIN_INFLUENCE_FACTOR + (1.0 - STYLE_MIN_INFLUENCE_FACTOR) * closeness,
        STYLE_MIN_INFLUENCE_FACTOR,
        1.0,
    )


def apply_style_influence_gate(effect: StyleMatchupEffect, influence: float) -> StyleMatchupEffect:
    influence = clamp(influence, 0.0, 1.0)
    if influence >= 0.999:
        return effect
    reasons = (f"influence_gate:{influence:.2f}", *effect.reasons)
    return StyleMatchupEffect(
        edge=effect.edge * influence,
        total_goal_multiplier=1.0 + (effect.total_goal_multiplier - 1.0) * influence,
        reasons=reasons,
    )


def profile_style_features(
    *,
    style: str,
    goals_for: float | None = None,
    goals_against: float | None = None,
    clean_sheet_rate: float | None = None,
    multi_goal_rate: float | None = None,
    conceded_multi_rate: float | None = None,
    high_total_goal_rate: float | None = None,
    both_score_rate: float | None = None,
) -> frozenset[str]:
    features: set[str] = set()
    primary = STYLE_FEATURES.get(style)
    if primary:
        features.add(primary)

    if goals_for is not None and (goals_for >= 1.95 or (multi_goal_rate or 0.0) >= 0.55):
        features.add("attack_high")
    if goals_for is not None and goals_for <= 1.25:
        features.add("attack_low")
    if goals_against is not None and (
        goals_against <= 0.78 or (clean_sheet_rate or 0.0) >= 0.52
    ):
        features.add("defense_strong")
    if goals_against is not None and (
        goals_against >= 1.05 or (conceded_multi_rate or 0.0) >= 0.27
    ):
        features.add("defense_fragility")
    if high_total_goal_rate is not None and (
        high_total_goal_rate >= 0.55 or (both_score_rate or 0.0) >= 0.50
    ):
        features.add("high_event")
    if high_total_goal_rate is not None and (
        high_total_goal_rate <= 0.42 and (both_score_rate or 0.0) <= 0.40
    ):
        features.add("low_event")
    if goals_for is not None and goals_against is not None and goals_for >= 1.70 and goals_against <= 0.90:
        features.add("control")
    if (
        goals_against is not None
        and clean_sheet_rate is not None
        and conceded_multi_rate is not None
        and goals_against <= 0.82
        and clean_sheet_rate >= 0.48
        and conceded_multi_rate <= 0.20
    ):
        features.add("suppression")
    if (
        goals_for is not None
        and both_score_rate is not None
        and goals_for >= 1.55
        and both_score_rate >= 0.43
    ):
        features.add("transition_route")

    return frozenset(sorted(features))


def profile_row_style_features(row: dict) -> frozenset[str]:
    return profile_style_features(
        style=row["style"],
        goals_for=float(row["weighted_goals_for"]),
        goals_against=float(row["weighted_goals_against"]),
        clean_sheet_rate=float(row["clean_sheet_rate"]),
        multi_goal_rate=float(row["multi_goal_rate"]),
        conceded_multi_rate=float(row["conceded_multi_rate"]),
        high_total_goal_rate=float(row["high_total_goal_rate"]),
        both_score_rate=float(row["both_score_rate"]),
    )


def prediction_row_style_features(row: dict, side: str) -> frozenset[str]:
    existing = row.get(f"style_features_{side}", "")
    if existing:
        return frozenset(feature for feature in existing.split(";") if feature)
    return profile_style_features(style=row[f"style_{side}"])


def team_shape_style_features(labels: str | frozenset[str]) -> frozenset[str]:
    if isinstance(labels, str):
        raw_labels = {label for label in labels.split(";") if label}
    else:
        raw_labels = set(labels)
    return frozenset(
        feature
        for label in raw_labels
        if (feature := TEAM_SHAPE_FEATURES.get(label)) is not None
    )


@lru_cache(maxsize=1)
def load_style_edges(path: Path = STYLE_MATCHUP_EDGES_CSV) -> dict[tuple[str, str], StyleEdge]:
    if not path.exists():
        return {}
    edges: dict[tuple[str, str], StyleEdge] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            edge = StyleEdge(
                feature_a=row["feature_a"],
                feature_b=row["feature_b"],
                weighted_matches=float(row["weighted_matches"]),
                residual_points_share_a=float(row["residual_points_share_a"]),
                shrunk_residual=float(row["shrunk_residual"]),
                total_goal_multiplier=float(row["total_goal_multiplier"]),
            )
            edges[(edge.feature_a, edge.feature_b)] = edge
    return edges


def strongest_features(features: frozenset[str]) -> tuple[str, ...]:
    priority = [
        "suppression",
        "control",
        "defense_strong",
        "defense_resistance",
        "transition_route",
        "attack_high",
        "high_event",
        "low_event",
        "style_complete",
        "style_attacking",
        "style_defensive",
        "style_open",
        "style_balanced",
        "style_low_efficiency",
        "attack_low",
        "defense_fragility",
    ]
    ordered = [feature for feature in priority if feature in features]
    ordered.extend(sorted(feature for feature in features if feature not in set(ordered)))
    return tuple(ordered[:MAX_PAIR_FEATURES])


def style_matchup_effect(
    features_a: frozenset[str],
    features_b: frozenset[str],
    edges: dict[tuple[str, str], StyleEdge] | None = None,
) -> StyleMatchupEffect:
    edge_table = load_style_edges() if edges is None else edges
    if not edge_table:
        return StyleMatchupEffect(0.0, 1.0, ())

    weighted_edge = 0.0
    total_weight = 0.0
    weighted_total = 0.0
    reasons: list[tuple[float, str]] = []
    for feature_a in strongest_features(features_a):
        for feature_b in strongest_features(features_b):
            edge = edge_table.get((feature_a, feature_b))
            if edge is None:
                continue
            if edge.weighted_matches < MIN_EDGE_WEIGHT or abs(edge.shrunk_residual) < MIN_ABS_EDGE:
                continue
            weight = min(120.0, edge.weighted_matches) * abs(edge.shrunk_residual)
            weighted_edge += edge.shrunk_residual * weight
            weighted_total += edge.total_goal_multiplier * weight
            total_weight += weight
            reasons.append(
                (
                    abs(edge.shrunk_residual),
                    f"{feature_a}>{feature_b}:{edge.shrunk_residual:+.3f}/{edge.weighted_matches:.0f}",
                )
            )

    contrast_edge = style_contrast_edge(features_a, features_b)
    contrast_total_multiplier = style_contrast_total_multiplier(features_a, features_b)
    contrast_reasons = style_contrast_reasons(features_a, features_b)

    if total_weight <= 0:
        return StyleMatchupEffect(
            clamp(contrast_edge, -0.090, 0.090),
            contrast_total_multiplier,
            tuple(contrast_reasons),
        )

    raw_edge = weighted_edge / total_weight
    edge = clamp(raw_edge + contrast_edge, -0.090, 0.090)
    raw_total_multiplier = weighted_total / total_weight
    total_multiplier = clamp((1.0 + (raw_total_multiplier - 1.0) * 0.20) * contrast_total_multiplier, 0.90, 1.03)
    top_reasons = tuple(contrast_reasons + [reason for _, reason in sorted(reasons, reverse=True)[:4]])
    return StyleMatchupEffect(edge, total_multiplier, top_reasons)


def style_contrast_edge(features_a: frozenset[str], features_b: frozenset[str]) -> float:
    edge = 0.0
    a_attack = bool(features_a & {"style_attacking", "attack_high", "high_event"})
    b_attack = bool(features_b & {"style_attacking", "attack_high", "high_event"})
    if "suppression" in features_a and "suppression" not in features_b and b_attack:
        edge += 0.075
    if "suppression" in features_b and "suppression" not in features_a and a_attack:
        edge -= 0.075

    if "transition_route" in features_a and "defense_fragility" in features_b:
        edge += 0.025
    if "transition_route" in features_b and "defense_fragility" in features_a:
        edge -= 0.025

    if "low_event" in features_a and b_attack and "low_event" not in features_b:
        edge += 0.018
    if "low_event" in features_b and a_attack and "low_event" not in features_a:
        edge -= 0.018

    return edge


def style_contrast_total_multiplier(features_a: frozenset[str], features_b: frozenset[str]) -> float:
    a_attack = bool(features_a & {"style_attacking", "attack_high", "high_event"})
    b_attack = bool(features_b & {"style_attacking", "attack_high", "high_event"})
    if (
        ("suppression" in features_a and "suppression" not in features_b and b_attack)
        or ("suppression" in features_b and "suppression" not in features_a and a_attack)
    ):
        return 0.92
    if (
        ("transition_route" in features_a and "defense_fragility" in features_b)
        or ("transition_route" in features_b and "defense_fragility" in features_a)
    ):
        return 1.03
    return 1.0


def style_contrast_reasons(features_a: frozenset[str], features_b: frozenset[str]) -> list[str]:
    reasons: list[str] = []
    a_attack = bool(features_a & {"style_attacking", "attack_high", "high_event"})
    b_attack = bool(features_b & {"style_attacking", "attack_high", "high_event"})
    if "suppression" in features_a and "suppression" not in features_b and b_attack:
        reasons.append("profile_contrast:A_suppression_vs_B_attack:+0.075")
    if "suppression" in features_b and "suppression" not in features_a and a_attack:
        reasons.append("profile_contrast:B_suppression_vs_A_attack:-0.075")
    if "transition_route" in features_a and "defense_fragility" in features_b:
        reasons.append("profile_contrast:A_transition_vs_B_fragility:+0.025")
    if "transition_route" in features_b and "defense_fragility" in features_a:
        reasons.append("profile_contrast:B_transition_vs_A_fragility:-0.025")
    if "low_event" in features_a and b_attack and "low_event" not in features_b:
        reasons.append("profile_contrast:A_low_event_vs_B_attack:+0.018")
    if "low_event" in features_b and a_attack and "low_event" not in features_a:
        reasons.append("profile_contrast:B_low_event_vs_A_attack:-0.018")
    return reasons


def write_feature_snapshot(
    rows: list[dict],
    output_path: Path = OUTPUT_DIR / "team_style_features_2026.csv",
) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        fieldnames = ["team", "style", "features"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["team"]):
            writer.writerow(
                {
                    "team": row["team"],
                    "style": row["style"],
                    "features": ";".join(profile_row_style_features(row)),
                }
            )
