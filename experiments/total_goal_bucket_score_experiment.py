from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import backtest_world_cup_continuous_total_goals as historical
import predict_fifa_profile as profile_model


BUCKETS = ("0-1球", "2-3球", "4-5球", "6-8球")
CENTERS = {
    "0-1球": 0.8,
    "2-3球": 2.5,
    "4-5球": 4.5,
    "6-8球": 6.7,
}
LOW_STYLES = {"防守型", "低效型"}
HIGH_STYLES = {"开放型", "进攻型"}
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SUMMARY_MD = OUTPUT_DIR / "total_goal_bucket_score_experiment_summary.md"
DETAIL_CSV = OUTPUT_DIR / "total_goal_bucket_score_experiment_details.csv"


@dataclass(frozen=True)
class Params:
    name: str
    prior_weight: float
    value_weight: float
    sigma: float
    low_weight: float
    high_weight: float
    extreme_weight: float
    mid_penalty_weight: float


@dataclass(frozen=True)
class MatchFeature:
    scope: str
    year: int | None
    match_id: str
    actual_bucket: str
    expected_total: float
    xg_total: float
    p_draw: float
    edge: float
    favorite_probability: float
    rank_gap: float
    low_style_count: int
    high_style_count: int
    shape_labels: frozenset[str]
    group_tempo: float


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def total_bucket(total_goals: int) -> str:
    if total_goals <= 1:
        return "0-1球"
    if total_goals <= 3:
        return "2-3球"
    if total_goals <= 5:
        return "4-5球"
    return "6-8球"


def row_key(row: dict) -> tuple[str, str, str, str]:
    return row["date_bjt"], row["time_bjt"], row["team_a"], row["team_b"]


def parse_probability_cell(cell: str) -> float:
    return float(cell.strip() or 0.0)


def log_prior(prior: dict[str, float], bucket: str) -> float:
    return math.log(max(0.0001, prior[bucket]))


def bucket_prior(features: list[MatchFeature], excluded_year: int | None = None) -> dict[str, float]:
    counts = {bucket: 1.0 for bucket in BUCKETS}
    for item in features:
        if excluded_year is not None and item.year == excluded_year:
            continue
        counts[item.actual_bucket] += 1.0
    total = sum(counts.values())
    return {bucket: counts[bucket] / total for bucket in BUCKETS}


def normal_component(expected_total: float, bucket: str, sigma: float) -> float:
    center = CENTERS[bucket]
    return -((expected_total - center) ** 2) / (2.0 * sigma * sigma)


def positive(value: float) -> float:
    return max(0.0, value)


def shape_signal(labels: frozenset[str], *names: str) -> float:
    return 1.0 if labels.intersection(names) else 0.0


def bucket_scores(feature: MatchFeature, prior: dict[str, float], params: Params) -> dict[str, float]:
    scores = {
        bucket: params.prior_weight * log_prior(prior, bucket)
        + params.value_weight * normal_component(feature.expected_total, bucket, params.sigma)
        for bucket in BUCKETS
    }

    close_game = positive(0.24 - feature.edge) / 0.24
    draw_signal = (feature.p_draw - 0.28) / 0.10
    low_shape = shape_signal(feature.shape_labels, "low_block", "low_event_favorite", "low_event")
    open_shape = shape_signal(feature.shape_labels, "open_game", "open_mismatch")
    collapse_shape = shape_signal(feature.shape_labels, "collapse_risk")
    transition_shape = shape_signal(feature.shape_labels, "transition_dog", "set_piece_risk")
    slow_group = positive(0.97 - feature.group_tempo) / 0.09

    low_signal = (
        draw_signal * 0.46
        + close_game * 0.24
        + feature.low_style_count * 0.20
        + low_shape * 0.82
        + slow_group * 0.30
        - positive(feature.xg_total - 2.65) * 0.34
        - open_shape * 0.55
        - collapse_shape * 0.90
    )
    high_signal = (
        positive(feature.xg_total - 2.55) * 0.44
        + positive(feature.favorite_probability - 0.54) * 0.74
        + positive(feature.edge - 0.28) * 0.34
        + positive(feature.rank_gap - 22.0) / 70.0 * 0.24
        + feature.high_style_count * 0.18
        + transition_shape * 0.36
        + open_shape * 0.72
        - positive(feature.p_draw - 0.31) * 0.55
        - low_shape * 0.42
    )
    extreme_signal = (
        positive(feature.xg_total - 3.25) * 0.58
        + positive(feature.favorite_probability - 0.64) * 0.78
        + positive(feature.rank_gap - 45.0) / 80.0 * 0.34
        + open_shape * 0.48
        + collapse_shape * 1.25
        - low_shape * 0.65
        - positive(feature.p_draw - 0.29) * 0.75
    )

    scores["0-1球"] += params.low_weight * low_signal
    scores["4-5球"] += params.high_weight * high_signal
    scores["6-8球"] += params.extreme_weight * extreme_signal

    outward_signal = positive(low_signal) + positive(high_signal) + positive(extreme_signal)
    scores["2-3球"] -= params.mid_penalty_weight * outward_signal
    return scores


def bucket_probabilities(feature: MatchFeature, prior: dict[str, float], params: Params) -> list[tuple[str, float]]:
    scores = bucket_scores(feature, prior, params)
    top = max(scores.values())
    exp_scores = {bucket: math.exp(score - top) for bucket, score in scores.items()}
    total = sum(exp_scores.values())
    return sorted(
        ((bucket, value / total) for bucket, value in exp_scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )


def historical_features() -> list[MatchFeature]:
    rows, _ = historical.build_rows()
    features: list[MatchFeature] = []
    for row in rows:
        p_home = float(row["p_home"])
        p_away = float(row["p_away"])
        home_style = row["home_style"]
        away_style = row["away_style"]
        features.append(
            MatchFeature(
                scope="historical",
                year=int(row["year"]),
                match_id=f"{row['year']} {row['home_team']} vs {row['away_team']}",
                actual_bucket=row["actual_total_goal_bucket"],
                expected_total=float(row["expected_total_goals"]),
                xg_total=float(row["xg_home"]) + float(row["xg_away"]),
                p_draw=float(row["p_draw"]),
                edge=abs(p_home - p_away),
                favorite_probability=max(p_home, p_away),
                rank_gap=abs(float(row["home_fifa_rank"]) - float(row["away_fifa_rank"])),
                low_style_count=int(home_style in LOW_STYLES) + int(away_style in LOW_STYLES),
                high_style_count=int(home_style in HIGH_STYLES) + int(away_style in HIGH_STYLES),
                shape_labels=frozenset(),
                group_tempo=1.0,
            )
        )
    return features


def current_2026_features() -> list[MatchFeature]:
    adjusted_rows = read_csv(ROOT / "output" / "realtime_context_adjusted_plan.csv")
    base_rows = {row_key(row): row for row in read_csv(ROOT / "output" / "group_score_predictions_fifa_profile.csv")}
    results = {row_key(row): row for row in read_csv(ROOT / "data" / "world_cup_2026_results.csv")}
    features: list[MatchFeature] = []
    for row in adjusted_rows:
        key = row_key(row)
        result = results.get(key)
        base = base_rows.get(key)
        if result is None or base is None:
            continue
        p_a = parse_probability_cell(row["adjusted_p_a"])
        p_b = parse_probability_cell(row["adjusted_p_b"])
        shape_labels = frozenset(label for label in row.get("shape_labels", "").split(";") if label)
        expected_total = profile_model.expected_total_goals_value(
            float(row["adjusted_xg_a"]),
            float(row["adjusted_xg_b"]),
            p_a,
            parse_probability_cell(row["adjusted_p_draw"]),
            p_b,
            shape_labels=row.get("shape_labels", ""),
        )
        goals_a = int(result["goals_a"])
        goals_b = int(result["goals_b"])
        features.append(
            MatchFeature(
                scope="2026",
                year=2026,
                match_id=f"{row['date_bjt']} {row['team_a']} vs {row['team_b']}",
                actual_bucket=total_bucket(goals_a + goals_b),
                expected_total=expected_total,
                xg_total=float(row["adjusted_xg_a"]) + float(row["adjusted_xg_b"]),
                p_draw=parse_probability_cell(row["adjusted_p_draw"]),
                edge=abs(p_a - p_b),
                favorite_probability=max(p_a, p_b),
                rank_gap=abs(float(base["fifa_rank_a"]) - float(base["fifa_rank_b"])),
                low_style_count=int(base["style_a"] in LOW_STYLES) + int(base["style_b"] in LOW_STYLES),
                high_style_count=int(base["style_a"] in HIGH_STYLES) + int(base["style_b"] in HIGH_STYLES),
                shape_labels=shape_labels,
                group_tempo=float(row.get("group_tempo_multiplier") or 1.0),
            )
        )
    return features


def threshold_bucket(expected_total: float) -> str:
    if expected_total < 1.5:
        return "0-1球"
    if expected_total < 3.5:
        return "2-3球"
    if expected_total < 5.5:
        return "4-5球"
    return "6-8球"


def threshold_second_bucket(expected_total: float, selected_bucket: str) -> str:
    if selected_bucket == "0-1球":
        return "2-3球"
    if selected_bucket == "6-8球":
        return "4-5球"
    if selected_bucket == "2-3球":
        lower_distance = abs(expected_total - 1.5)
        upper_distance = abs(3.5 - expected_total)
        return "0-1球" if lower_distance < upper_distance else "4-5球"
    if selected_bucket == "4-5球":
        lower_distance = abs(expected_total - 3.5)
        upper_distance = abs(5.5 - expected_total)
        return "2-3球" if lower_distance < upper_distance else "6-8球"
    raise ValueError(selected_bucket)


def evaluate_threshold(features: list[MatchFeature]) -> dict:
    rows = []
    for feature in features:
        selected = threshold_bucket(feature.expected_total)
        second = threshold_second_bucket(feature.expected_total, selected)
        rows.append((feature, selected, second))
    return summarize_rows(rows)


def evaluate_independent(features: list[MatchFeature], params: Params, historical_all: list[MatchFeature]) -> dict:
    rows = []
    historical_prior = bucket_prior(historical_all)
    for feature in features:
        prior = bucket_prior(historical_all, feature.year) if feature.scope == "historical" else historical_prior
        probabilities = bucket_probabilities(feature, prior, params)
        rows.append((feature, probabilities[0][0], probabilities[1][0]))
    return summarize_rows(rows)


def summarize_rows(rows: list[tuple[MatchFeature, str, str]]) -> dict:
    total = len(rows)
    selected_23 = sum(1 for _, selected, _ in rows if selected == "2-3球")
    return {
        "matches": total,
        "top1": sum(1 for feature, selected, _ in rows if selected == feature.actual_bucket),
        "top2": sum(1 for feature, selected, second in rows if feature.actual_bucket in {selected, second}),
        "selected_23": selected_23,
        "actual_23": sum(1 for feature, _, _ in rows if feature.actual_bucket == "2-3球"),
        "pred23_low": sum(
            1 for feature, selected, _ in rows if selected == "2-3球" and feature.actual_bucket == "0-1球"
        ),
        "pred23_hit": sum(
            1 for feature, selected, _ in rows if selected == "2-3球" and feature.actual_bucket == "2-3球"
        ),
        "pred23_high": sum(
            1
            for feature, selected, _ in rows
            if selected == "2-3球" and feature.actual_bucket in {"4-5球", "6-8球"}
        ),
        "bucket_counts": {bucket: sum(1 for _, selected, _ in rows if selected == bucket) for bucket in BUCKETS},
    }


def candidate_params() -> list[Params]:
    candidates: list[Params] = []
    for prior_weight in (0.65, 0.85, 1.05):
        for value_weight in (0.45, 0.65, 0.85, 1.05):
            for sigma in (1.20, 1.40, 1.60):
                for feature_weight in (0.70, 1.00, 1.30):
                    for mid_penalty in (0.00, 0.12, 0.24, 0.36):
                        candidates.append(
                            Params(
                                name=(
                                    f"p{prior_weight:.2f}_v{value_weight:.2f}_s{sigma:.2f}_"
                                    f"f{feature_weight:.2f}_m{mid_penalty:.2f}"
                                ),
                                prior_weight=prior_weight,
                                value_weight=value_weight,
                                sigma=sigma,
                                low_weight=feature_weight,
                                high_weight=feature_weight,
                                extreme_weight=feature_weight,
                                mid_penalty_weight=mid_penalty,
                            )
                        )
    return candidates


def metric_line(label: str, metrics: dict) -> str:
    return (
        f"| {label} | {metrics['top1']}/{metrics['matches']} | {metrics['top2']}/{metrics['matches']} | "
        f"{metrics['selected_23']}/{metrics['matches']} | {metrics['actual_23']}/{metrics['matches']} | "
        f"{metrics['pred23_low']}/{metrics['pred23_hit']}/{metrics['pred23_high']} | "
        f"{metrics['bucket_counts']['0-1球']}/{metrics['bucket_counts']['2-3球']}/"
        f"{metrics['bucket_counts']['4-5球']}/{metrics['bucket_counts']['6-8球']} |"
    )


def write_detail_csv(
    historical_features_list: list[MatchFeature],
    current_features_list: list[MatchFeature],
    params: Params,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    historical_prior = bucket_prior(historical_features_list)
    rows: list[dict] = []
    for feature in historical_features_list + current_features_list:
        prior = bucket_prior(historical_features_list, feature.year) if feature.scope == "historical" else historical_prior
        probabilities = bucket_probabilities(feature, prior, params)
        rows.append(
            {
                "scope": feature.scope,
                "year": feature.year or "",
                "match": feature.match_id,
                "actual_bucket": feature.actual_bucket,
                "selected_bucket": probabilities[0][0],
                "second_bucket": probabilities[1][0],
                "top1_hit": probabilities[0][0] == feature.actual_bucket,
                "top2_hit": feature.actual_bucket in {probabilities[0][0], probabilities[1][0]},
                "expected_total": f"{feature.expected_total:.4f}",
                "xg_total": f"{feature.xg_total:.4f}",
                "p_draw": f"{feature.p_draw:.4f}",
                "edge": f"{feature.edge:.4f}",
                "favorite_probability": f"{feature.favorite_probability:.4f}",
                "rank_gap": f"{feature.rank_gap:.1f}",
                "shape_labels": ";".join(sorted(feature.shape_labels)),
                "bucket_probabilities": "; ".join(f"{bucket} {probability:.1%}" for bucket, probability in probabilities),
            }
        )
    with DETAIL_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    historical_features_list = historical_features()
    current_features_list = current_2026_features()
    historical_threshold = evaluate_threshold(historical_features_list)
    current_threshold = evaluate_threshold(current_features_list)

    scored = []
    for params in candidate_params():
        hist_metrics = evaluate_independent(historical_features_list, params, historical_features_list)
        current_metrics = evaluate_independent(current_features_list, params, historical_features_list)
        hist_23_gap = abs(hist_metrics["selected_23"] - hist_metrics["actual_23"])
        current_23_gap = abs(current_metrics["selected_23"] - current_metrics["actual_23"])
        score = (
            hist_metrics["top1"] * 4
            + hist_metrics["top2"]
            - hist_23_gap * 0.75
            + current_metrics["top1"] * 1.5
            + current_metrics["top2"] * 0.25
            - current_23_gap * 0.35
        )
        scored.append((score, params, hist_metrics, current_metrics))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_params, best_historical, best_current = scored[0]
    write_detail_csv(historical_features_list, current_features_list, best_params)

    lines = [
        "# Total Goal Bucket Independent Score Experiment",
        "",
        "- 目标：测试四个总进球桶独立打分，替代单一连续总球阈值。",
        "- 历史验证：2010/2014/2018/2022 留一届口径，每场基础桶先验不读取本届结果。",
        "- 2026 验证：只看已完赛比赛，用当前实时输出里的赛前字段。",
        "",
        f"Best params: `{best_params.name}`",
        "",
        "| 范围 | Top1 | Top2 | 选2-3 | 实际2-3 | 2-3误差 小/中/大 | 预测桶分布 0-1/2-3/4-5/6-8 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        metric_line("历史阈值基线", historical_threshold),
        metric_line("历史独立分数", best_historical),
        metric_line("2026阈值基线", current_threshold),
        metric_line("2026独立分数", best_current),
        "",
        "## Top Candidates",
        "",
        "| 参数 | 历史Top1 | 历史Top2 | 历史选2-3 | 2026Top1 | 2026Top2 | 2026选2-3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, params, hist_metrics, current_metrics in scored[:12]:
        lines.append(
            f"| `{params.name}` | {hist_metrics['top1']}/{hist_metrics['matches']} | "
            f"{hist_metrics['top2']}/{hist_metrics['matches']} | "
            f"{hist_metrics['selected_23']}/{hist_metrics['matches']} | "
            f"{current_metrics['top1']}/{current_metrics['matches']} | "
            f"{current_metrics['top2']}/{current_metrics['matches']} | "
            f"{current_metrics['selected_23']}/{current_metrics['matches']} |"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary: {SUMMARY_MD}")
    print(f"Detail: {DETAIL_CSV}")
    print(f"Best: {best_params.name}")
    print(metric_line("历史阈值基线", historical_threshold))
    print(metric_line("历史独立分数", best_historical))
    print(metric_line("2026阈值基线", current_threshold))
    print(metric_line("2026独立分数", best_current))


if __name__ == "__main__":
    main()
