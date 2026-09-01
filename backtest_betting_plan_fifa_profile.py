from __future__ import annotations

import csv
import sys
from collections import Counter

from betting_plan_fifa_profile import MAX_ITEMS, MAX_TOTAL_GOAL_BUCKET_ITEMS, plan_items, target_item_count
from profiles import OUTPUT_DIR


SOURCE_CSV = OUTPUT_DIR / "world_cup_fifa_profile_score_backtest_matches.csv"
MATCH_CSV = OUTPUT_DIR / "world_cup_betting_plan_backtest_matches.csv"
SUMMARY_MD = OUTPUT_DIR / "world_cup_betting_plan_backtest_summary.md"


def pct(value: float) -> str:
    return f"{value:.1%}"


def score_outcome(score: str) -> str:
    home, away = (int(part) for part in score.split("-"))
    if home > away:
        return "home"
    if home < away:
        return "away"
    return "draw"


def item_hits(row: dict, items: list[str]) -> tuple[bool, bool, bool, bool]:
    actual_total = int(row["actual_total_goals"])
    actual_score = row["actual_score"]
    actual_outcome = row["actual_outcome"]
    outcome_hit = False
    total_hit = False
    score_hit = False
    for item in items:
        if item.startswith("总进球："):
            bucket = item.removeprefix("总进球：")
            if bucket == "0-1球":
                total_hit = total_hit or actual_total <= 1
            elif bucket == "2-3球":
                total_hit = total_hit or 2 <= actual_total <= 3
            elif bucket == "4-5球":
                total_hit = total_hit or 4 <= actual_total <= 5
            elif bucket == "6-8球":
                total_hit = total_hit or 6 <= actual_total <= 8
            else:
                raise ValueError(f"cannot parse total-goal bucket item: {item}")
        elif item.startswith("比分"):
            _, score = item.split("：", 1)
            score_hit = score_hit or actual_score == score
        else:
            raise ValueError(f"unknown betting item: {item}")
    return outcome_hit or total_hit or score_hit, outcome_hit, total_hit, score_hit


def to_plan_row(row: dict) -> dict:
    return {
        "team_a": row["home_team"],
        "team_b": row["away_team"],
        "predicted_outcome": {"home": "A", "draw": "D", "away": "B"}[row["predicted_outcome"]],
        "p_a": row["p_home"],
        "p_draw": row["p_draw"],
        "p_b": row["p_away"],
        "uncertainty_score": row["uncertainty_score"],
        "top_total_goal_buckets": row["top_total_goal_buckets"],
        "recommended_score": row["recommended_score"],
        "recommended_score_probability": row["recommended_score_probability"],
        "bucket_primary_score": row["bucket_primary_score"],
        "bucket_primary_score_probability": row["bucket_primary_score_probability"],
        "bucket_complement_score": row["bucket_complement_score"],
        "bucket_complement_score_probability": row["bucket_complement_score_probability"],
        "aggressive_score": row["bucket_complement_score"],
        "aggressive_score_probability": row["bucket_complement_score_probability"],
        "market_value_score": row["recommended_score"],
        "total_constrained_score": row["total_constrained_score"],
        "total_constrained_score_probability": row["total_constrained_score_probability"],
        "top_scores": row["top_scores"],
    }


def load_rows() -> list[dict]:
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def build_rows() -> list[dict]:
    output = []
    for row in load_rows():
        items = plan_items(to_plan_row(row))
        limit = target_item_count(to_plan_row(row))
        if len(items) > limit:
            raise RuntimeError(f"too many items for {row['year']} {row['home_team']} vs {row['away_team']}: {len(items)}")
        any_hit, outcome_hit, total_hit, score_hit = item_hits(row, items)
        output.append(
            {
                "year": row["year"],
                "date": row["date"],
                "stage": row["stage"],
                "match": f"{row['home_team']} vs {row['away_team']}",
                "actual_score": row["actual_score"],
                "actual_total_goals": row["actual_total_goals"],
                "risk_label": row["risk_label"],
                "item_limit": limit,
                "items_count": len(items),
                "items": "；".join(items),
                "any_hit": any_hit,
                "outcome_hit": outcome_hit,
                "total_hit": total_hit,
                "score_hit": score_hit,
            }
        )
    return output


def accuracy(rows: list[dict], key: str) -> float:
    if not rows:
        raise ValueError("cannot compute accuracy of empty rows")
    return sum(1 for row in rows if row[key]) / len(rows)


def summary_row(label: str, rows: list[dict]) -> str:
    return (
        f"| {label} | {len(rows)} | "
        f"{sum(1 for row in rows if row['any_hit'])} | {pct(accuracy(rows, 'any_hit'))} | "
        f"{sum(1 for row in rows if row['total_hit'])} | {pct(accuracy(rows, 'total_hit'))} | "
        f"{sum(1 for row in rows if row['score_hit'])} | {pct(accuracy(rows, 'score_hit'))} | "
        f"{sum(int(row['items_count']) for row in rows) / len(rows):.2f} |"
    )


def write_match_csv(rows: list[dict]) -> None:
    fields = [
        "year",
        "date",
        "stage",
        "match",
        "actual_score",
        "actual_total_goals",
        "risk_label",
        "item_limit",
        "items_count",
        "items",
        "any_hit",
        "outcome_hit",
        "total_hit",
        "score_hit",
    ]
    with MATCH_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict]) -> None:
    risk_counts = Counter(row["risk_label"] for row in rows)
    lines = [
        "# 世界杯彩票组合方案回测",
        "",
        "- 覆盖年份：2010、2014、2018、2022。",
        "- 严格历史回测：只使用赛前 FIFA 排名、赛前 10 年画像、留一届总进球基准。",
        "- 不读取实时上下文、比赛形态、赛后技术统计、赛后媒体评论。",
        "- 每场固定 4 项：1 个 Top1 总进球桶 + 3 个赛前比分。",
        f"- 总进球桶固定 {MAX_TOTAL_GOAL_BUCKET_ITEMS} 个；总项数固定 {MAX_ITEMS}。",
        "- 命中定义：方案内任一项命中即算该场命中。",
        "- 这只是历史覆盖率，不代表收益率。",
        "",
        "## 命中率",
        "",
        "| 范围 | 场次 | 任一项命中 | 命中率 | 总进球桶命中 | 命中率 | 任一比分命中 | 命中率 | 平均项数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        summary_row("全部", rows),
    ]
    for year in sorted({row["year"] for row in rows}):
        lines.append(summary_row(year, [row for row in rows if row["year"] == year]))
    lines.append(summary_row("小组赛", [row for row in rows if row["stage"] == "group"]))
    lines.append(summary_row("淘汰赛", [row for row in rows if row["stage"] == "knockout"]))

    lines.extend(
        [
            "",
            "## 按风险",
            "",
            "| 风险 | 场次 | 任一项命中率 | 平均项数 |",
            "|---|---:|---:|---:|",
        ]
    )
    for label in ["低", "中", "中高", "高"]:
        bucket = [row for row in rows if row["risk_label"] == label]
        if bucket:
            lines.append(
                f"| {label} | {risk_counts[label]} | {pct(accuracy(bucket, 'any_hit'))} | "
                f"{sum(int(row['items_count']) for row in bucket) / len(bucket):.2f} |"
            )

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    rows = build_rows()
    write_match_csv(rows)
    write_summary(rows)
    print(f"Matches: {MATCH_CSV}")
    print(f"Summary: {SUMMARY_MD}")
    print(f"Any-hit accuracy: {sum(1 for row in rows if row['any_hit'])}/{len(rows)} = {accuracy(rows, 'any_hit'):.1%}")


if __name__ == "__main__":
    main()
