from __future__ import annotations

import csv
import re
import sys

from prediction_rules import parse_score, total_goal_bucket
from profiles import OUTPUT_DIR

PREDICTIONS_CSV = OUTPUT_DIR / "group_score_predictions_fifa_profile.csv"
ADJUSTED_PREDICTIONS_CSV = OUTPUT_DIR / "realtime_context_adjusted_plan.csv"
PLAN_CSV = OUTPUT_DIR / "betting_plan_fifa_profile.csv"
PLAN_MD = OUTPUT_DIR / "betting_plan_fifa_profile.md"
MAX_TOTAL_GOAL_BUCKET_ITEMS = 1
MAX_ITEMS = 5


def parse_top_total_goal_buckets(value: str) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for part in value.split("; "):
        match = re.match(r"^(0-1球|2-3球|4-5球|6-8球) ([\d.]+)%$", part.strip())
        if not match:
            raise ValueError(f"cannot parse total-goal bucket item: {part}")
        items.append((match.group(1), float(match.group(2)) / 100.0))
    return items


def parse_top_scores(value: str) -> list[tuple[str, float]]:
    if not value:
        return []
    items: list[tuple[str, float]] = []
    for part in value.split("; "):
        match = re.match(r"^(\d+-\d+) ([\d.]+)%$", part.strip())
        if not match:
            raise ValueError(f"cannot parse score item: {part}")
        items.append((match.group(1), float(match.group(2)) / 100.0))
    return items


def score_total_goal_bucket(score: str) -> str:
    home, away = parse_score(score)
    return total_goal_bucket(home + away)


def target_item_count(row: dict) -> int:
    return MAX_ITEMS


def prediction_value(row: dict, adjusted_key: str, base_key: str) -> str:
    return row.get(adjusted_key) or row[base_key]


def top_total_goal_buckets_value(row: dict) -> str:
    return prediction_value(row, "adjusted_total_goals_top2", "top_total_goal_buckets")


def outcome_label(row: dict) -> str:
    if row["predicted_outcome"] == "A":
        return f"{row['team_a']}胜"
    if row["predicted_outcome"] == "B":
        return f"{row['team_b']}胜"
    if row["predicted_outcome"] == "D":
        return "平局"
    raise ValueError(f"unknown outcome: {row['predicted_outcome']}")


def plan_items(row: dict) -> list[str]:
    buckets = parse_top_total_goal_buckets(top_total_goal_buckets_value(row))
    items = [f"总进球：{bucket}" for bucket, _ in buckets[:MAX_TOTAL_GOAL_BUCKET_ITEMS]]
    score_1, score_2, score_3, score_4 = score_items(row)
    items.append(f"比分1：{score_1}")
    items.append(f"比分2：{score_2}")
    items.append(f"比分3：{score_3}")
    items.append(f"比分4：{score_4}")
    return items


def score_items(row: dict) -> tuple[str, str, str, str]:
    return (
        prediction_value(row, "adjusted_score_1_model", "bucket_primary_score"),
        prediction_value(row, "adjusted_score_2_aggressive_prediction", "aggressive_score"),
        prediction_value(row, "adjusted_score_3_market_value", "market_value_score"),
        prediction_value(row, "adjusted_score_4_upset", "upset_score"),
    )


def score_candidates(row: dict) -> list[str]:
    return list(score_items(row))


def score_item_probability(row: dict, score: str) -> float:
    if score == row.get("bucket_primary_score"):
        return float(row["bucket_primary_score_probability"])
    if score == row.get("recommended_score"):
        return float(row["recommended_score_probability"])
    if score == row.get("aggressive_score"):
        return float(row["aggressive_score_probability"])
    if score == row.get("bucket_complement_score"):
        return float(row["bucket_complement_score_probability"])
    if score == row.get("total_constrained_score"):
        return float(row["total_constrained_score_probability"])
    if score == row.get("market_value_score"):
        return 0.0
    if score == row.get("upset_score"):
        return float(row.get("upset_score_probability", 0.0))
    if score in {
        row.get("adjusted_score_1_model"),
        row.get("adjusted_score_2_aggressive_prediction"),
        row.get("adjusted_score_3_market_value"),
        row.get("adjusted_score_4_upset"),
    }:
        return float(row.get("adjusted_score_4_upset_probability", 0.0)) if score == row.get("adjusted_score_4_upset") else 0.0
    for top_score, probability in parse_top_scores(row.get("top_scores", "")):
        if score == top_score:
            return probability
    raise ValueError(f"unknown score item for probability: {score}")


def combined_cover_probability(row: dict, items: list[str]) -> float:
    goal_probability = 0.0
    outcome_probability = 0.0
    buckets = dict(parse_top_total_goal_buckets(top_total_goal_buckets_value(row)))
    selected_buckets: set[str] = set()
    for item in items:
        if item.startswith("总进球："):
            bucket = item.removeprefix("总进球：")
            selected_buckets.add(bucket)
            goal_probability += buckets.get(bucket, 0.0)
        elif item.startswith("比分"):
            _, score = item.split("：", 1)
            if score_total_goal_bucket(score) not in selected_buckets:
                goal_probability += score_item_probability(row, score)
        else:
            raise ValueError(f"unknown betting item: {item}")

    return min(1.0, max(goal_probability, outcome_probability))


def prediction_key(row: dict) -> tuple[str, str, str, str]:
    return row["date_bjt"], row["time_bjt"], row["team_a"], row["team_b"]


def load_csv(path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_predictions() -> list[dict]:
    base_rows = load_csv(PREDICTIONS_CSV)
    if not ADJUSTED_PREDICTIONS_CSV.exists():
        return base_rows
    adjusted_by_key = {prediction_key(row): row for row in load_csv(ADJUSTED_PREDICTIONS_CSV)}
    return [{**row, **adjusted_by_key.get(prediction_key(row), {})} for row in base_rows]


def build_rows() -> list[dict]:
    rows = []
    for row in load_predictions():
        items = plan_items(row)
        top_total_goal_buckets = top_total_goal_buckets_value(row)
        score_1_model, score_2_aggressive, score_3_market, score_4_upset = score_items(row)
        rows.append(
            {
                "date_bjt": row["date_bjt"],
                "time_bjt": row["time_bjt"],
                "group": row["group"],
                "match": f"{row['team_a']} vs {row['team_b']}",
                "risk_label": row["risk_label"],
                "risk_reasons": row["risk_reasons"],
                "outcome_reference": outcome_label(row),
                "item_limit": target_item_count(row),
                "items_count": len(items),
                "model_cover_probability": combined_cover_probability(row, items),
                "items": "；".join(items),
                "score_1_model": score_1_model,
                "score_2_aggressive_prediction": score_2_aggressive,
                "score_3_market_value": score_3_market,
                "score_4_upset": score_4_upset,
                "p_home": row["p_a"],
                "p_draw": row["p_draw"],
                "p_away": row["p_b"],
                "top_total_goal_buckets": top_total_goal_buckets,
                "bucket_complement_score": row["bucket_complement_score"],
                "bucket_complement_score_probability": row["bucket_complement_score_probability"],
                "recommended_score": row["recommended_score"],
                "recommended_score_probability": row["recommended_score_probability"],
                "aggressive_score": row["aggressive_score"],
                "aggressive_score_probability": row["aggressive_score_probability"],
                "market_value_score": prediction_value(row, "adjusted_score_3_market_value", "market_value_score"),
                "upset_score": prediction_value(row, "adjusted_score_4_upset", "upset_score"),
                "upset_score_probability": row.get("adjusted_score_4_upset_probability") or row.get("upset_score_probability", ""),
                "market_value_a_eur_m": row["market_value_a_eur_m"],
                "market_value_b_eur_m": row["market_value_b_eur_m"],
                "bucket_primary_score": prediction_value(row, "adjusted_score_1_model", "bucket_primary_score"),
                "bucket_primary_score_probability": row["bucket_primary_score_probability"],
                "total_constrained_score": row["total_constrained_score"],
                "total_constrained_score_probability": row["total_constrained_score_probability"],
                "top_scores": row.get("top_scores", ""),
            }
        )
    return rows


def write_csv(rows: list[dict]) -> None:
    fields = [
        "date_bjt",
        "time_bjt",
        "group",
        "match",
        "risk_label",
        "risk_reasons",
        "outcome_reference",
        "item_limit",
        "items_count",
        "model_cover_probability",
        "items",
        "score_1_model",
        "score_2_aggressive_prediction",
        "score_3_market_value",
        "score_4_upset",
        "p_home",
        "p_draw",
        "p_away",
        "top_total_goal_buckets",
        "bucket_complement_score",
        "bucket_complement_score_probability",
        "recommended_score",
        "recommended_score_probability",
        "aggressive_score",
        "aggressive_score_probability",
        "market_value_score",
        "upset_score",
        "upset_score_probability",
        "market_value_a_eur_m",
        "market_value_b_eur_m",
        "bucket_primary_score",
        "bucket_primary_score_probability",
        "total_constrained_score",
        "total_constrained_score_probability",
        "top_scores",
    ]
    with PLAN_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "model_cover_probability": f"{row['model_cover_probability']:.6f}"})


def write_markdown(rows: list[dict]) -> None:
    lines = [
        "# 2026 世界杯小组赛彩票组合方案",
        "",
        "- 这是模型概率组合，不是收益承诺。",
        "- 胜负只做参考，不进入彩票项。",
        "- 每场固定 5 项：1 个全场总进球桶 + 4 个比分。",
        "- 比分1：模型输出分，落在 Top1 总进球桶。",
        "- 比分2：备选比分，落在 Top2 总进球桶，不保证大于模型分。",
        "- 比分3：身价预测分，只允许落在 Top1 或 Top2 总进球桶。",
        "- 比分4：爆冷/压缩分，热门胜率低于70%时覆盖平局或弱队小胜，强热门时改为压缩分。",
        "",
        "| 北京时间 | 组 | 比赛 | 胜负参考 | 风险 | 总进球 | 模型 | 备选 | 身价 | 爆冷 | 覆盖强度 |",
        "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {date} {time} | {group} | {match} | {outcome} | {risk} | {bucket} | {s1} | {s2} | {s3} | {s4} | {cover:.1%} |".format(
                date=row["date_bjt"],
                time=row["time_bjt"],
                group=row["group"],
                match=row["match"],
                outcome=row["outcome_reference"],
                risk=f"{row['risk_label']}：{row['risk_reasons']}",
                bucket=parse_top_total_goal_buckets(row["top_total_goal_buckets"])[0][0],
                s1=row["score_1_model"],
                s2=row["score_2_aggressive_prediction"],
                s3=row["score_3_market_value"],
                s4=row["score_4_upset"],
                cover=row["model_cover_probability"],
            )
        )
    PLAN_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    rows = build_rows()
    write_csv(rows)
    write_markdown(rows)
    print(f"CSV: {PLAN_CSV}")
    print(f"Markdown: {PLAN_MD}")
    for row in rows[:8]:
        print(f"{row['date_bjt']} {row['match']}: {row['items']}")


if __name__ == "__main__":
    main()
