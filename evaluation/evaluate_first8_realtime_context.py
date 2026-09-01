from __future__ import annotations

import csv
import sys
from datetime import date

from predict import Match, OUTPUT_DIR
from predict_fifa_profile import (
    load_fifa_rankings,
    load_club_cohesion,
    load_market_values,
    load_profiles,
    predict_match,
    profile_baselines,
    total_goal_bucket,
)
from realtime_context_adjusted_plan import apply_context, load_completed_matches, load_context, load_match_shapes


EVAL_CSV = OUTPUT_DIR / "first8_realtime_context_eval.csv"
EVAL_MD = OUTPUT_DIR / "first8_realtime_context_eval.md"

FIRST8 = [
    (Match("A", date(2026, 6, 11), "12:00", "Mexico", "South Africa", "Mexico City"), (2, 0)),
    (Match("A", date(2026, 6, 11), "18:00", "South Korea", "Czechia", "Zapopan"), (2, 1)),
    (Match("B", date(2026, 6, 12), "15:00", "Canada", "Bosnia and Herzegovina", "Toronto"), (1, 1)),
    (Match("D", date(2026, 6, 12), "21:00", "United States", "Paraguay", "Inglewood"), (4, 1)),
    (Match("B", date(2026, 6, 13), "15:00", "Qatar", "Switzerland", "Santa Clara"), (1, 1)),
    (Match("C", date(2026, 6, 13), "18:00", "Brazil", "Morocco", "East Rutherford"), (1, 1)),
    (Match("C", date(2026, 6, 13), "21:00", "Haiti", "Scotland", "Foxborough"), (0, 1)),
    (Match("D", date(2026, 6, 14), "00:00", "Australia", "Turkey", "Vancouver"), (2, 0)),
]


def outcome(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "A"
    if goals_a < goals_b:
        return "B"
    return "D"


def score_text(scores: list[str]) -> str:
    return " / ".join(scores)


def hit_text(value: bool) -> str:
    return "是" if value else "否"


def evaluate_rows() -> list[dict]:
    rankings = load_fifa_rankings()
    profiles = load_profiles()
    market_values = load_market_values()
    club_cohesion = load_club_cohesion()
    baselines = profile_baselines(list(profiles.values()))
    contexts = load_context()
    shapes = load_match_shapes()
    completed_matches = load_completed_matches()
    rows: list[dict] = []
    for match, actual in FIRST8:
        base = predict_match(match, rankings, profiles, baselines, market_values, club_cohesion)
        adjusted = apply_context(base, contexts, shapes, completed_matches)
        actual_score = f"{actual[0]}-{actual[1]}"
        actual_bucket = total_goal_bucket(actual[0] + actual[1])

        base_scores = [
            base["recommended_score"],
            base["aggressive_score"],
            base["market_value_score"],
        ]
        adjusted_scores = [
            adjusted["adjusted_score_1_model"],
            adjusted["adjusted_score_2_aggressive_prediction"],
            adjusted["adjusted_score_3_market_value"],
        ]

        base_total_hit = base["selected_total_goal_bucket"] == actual_bucket
        adjusted_total_hit = adjusted["adjusted_total_goal_bucket"] == actual_bucket
        base_score_hit = actual_score in base_scores
        adjusted_score_hit = actual_score in adjusted_scores
        wdl_hit = base["predicted_outcome"] == outcome(*actual)
        rows.append(
            {
                "date_bjt": base["date_bjt"],
                "match": f"{base['team_a']} vs {base['team_b']}",
                "actual": actual_score,
                "actual_total_bucket": actual_bucket,
                "wdl_reference": base["predicted_outcome"],
                "wdl_hit": wdl_hit,
                "base_total_bucket": base["selected_total_goal_bucket"],
                "adjusted_total_bucket": adjusted["adjusted_total_goal_bucket"],
                "base_scores": score_text(base_scores),
                "adjusted_scores": score_text(adjusted_scores),
                "base_total_hit": base_total_hit,
                "adjusted_total_hit": adjusted_total_hit,
                "base_score_hit": base_score_hit,
                "adjusted_score_hit": adjusted_score_hit,
                "base_covered": base_total_hit or base_score_hit,
                "adjusted_covered": adjusted_total_hit or adjusted_score_hit,
                "context_applied": adjusted["context_applied"],
            }
        )
    return rows


def count(rows: list[dict], field: str) -> int:
    return sum(1 for row in rows if row[field])


def write_csv(rows: list[dict]) -> None:
    fields = [
        "date_bjt",
        "match",
        "actual",
        "actual_total_bucket",
        "wdl_reference",
        "wdl_hit",
        "base_total_bucket",
        "adjusted_total_bucket",
        "base_scores",
        "adjusted_scores",
        "base_total_hit",
        "adjusted_total_hit",
        "base_score_hit",
        "adjusted_score_hit",
        "base_covered",
        "adjusted_covered",
        "context_applied",
    ]
    with EVAL_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict]) -> None:
    total = len(rows)
    lines = [
        "# 前8场赛后诊断",
        "",
        "- 该结果不能作为模型验证准确率。",
        "- 实时层可能包含赛后录入的形态和上下文，只用于解释偏差与更新后续模型。",
        "",
        "| 指标 | 基础模型 | 实时层后 |",
        "|---|---:|---:|",
        f"| 胜平负参考命中 | {count(rows, 'wdl_hit')}/{total} | {count(rows, 'wdl_hit')}/{total} |",
        f"| 总进球桶命中 | {count(rows, 'base_total_hit')}/{total} | {count(rows, 'adjusted_total_hit')}/{total} |",
        f"| 三个比分任一命中 | {count(rows, 'base_score_hit')}/{total} | {count(rows, 'adjusted_score_hit')}/{total} |",
        f"| 总进球桶或比分覆盖 | {count(rows, 'base_covered')}/{total} | {count(rows, 'adjusted_covered')}/{total} |",
        "",
        "| 比赛 | 实际 | 基础总球 | 实时总球 | 基础比分 | 实时比分 | 基础覆盖 | 实时覆盖 |",
        "|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {match} | {actual} | {base_bucket} | {adjusted_bucket} | {base_scores} | {adjusted_scores} | {base_hit} | {adjusted_hit} |".format(
                match=row["match"],
                actual=row["actual"],
                base_bucket=row["base_total_bucket"],
                adjusted_bucket=row["adjusted_total_bucket"],
                base_scores=row["base_scores"],
                adjusted_scores=row["adjusted_scores"],
                base_hit=hit_text(row["base_covered"]),
                adjusted_hit=hit_text(row["adjusted_covered"]),
            )
        )
    EVAL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    rows = evaluate_rows()
    write_csv(rows)
    write_markdown(rows)
    total = len(rows)
    print(f"CSV: {EVAL_CSV}")
    print(f"Markdown: {EVAL_MD}")
    print(
        "覆盖: "
        f"{count(rows, 'base_covered')}/{total} -> {count(rows, 'adjusted_covered')}/{total}; "
        f"比分: {count(rows, 'base_score_hit')}/{total} -> {count(rows, 'adjusted_score_hit')}/{total}; "
        f"总进球桶: {count(rows, 'base_total_hit')}/{total} -> {count(rows, 'adjusted_total_hit')}/{total}"
    )


if __name__ == "__main__":
    main()
