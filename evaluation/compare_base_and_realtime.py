from __future__ import annotations

from pathlib import Path

from evaluation.evaluate_finished_from_realtime_cache import (
    SCORE_COLUMNS,
    archive_for_match,
    evaluated_row,
    load_prediction_archive,
    load_results,
    pct,
    score_column_summary,
    summarize,
)

OUTPUT_MD = Path(__file__).resolve().parents[1] / "output" / "base_vs_realtime_evaluation_summary.md"


def base_prediction(archived: dict) -> dict:
    return {
        "predicted_outcome": archived["base_predicted_outcome"],
        "adjusted_total_goal_bucket": archived["base_selected_total_goal_bucket"],
        "adjusted_total_goals_top2": archived["base_top_total_goal_buckets"],
        "adjusted_score_1_model": archived["base_recommended_score"],
        "adjusted_score_2_aggressive_prediction": archived["base_aggressive_score"],
        "adjusted_score_3_market_value": archived["base_market_value_score"],
        "adjusted_score_4_upset": "",
    }


def comparison_rows() -> tuple[list[dict], list[dict]]:
    archived = load_prediction_archive()
    base_rows: list[dict] = []
    realtime_rows: list[dict] = []
    for result in load_results():
        found = archive_for_match(result, archived)
        if found is None:
            continue
        run, prediction = found
        base_rows.append(evaluated_row(result, run, base_prediction(prediction)))
        realtime_rows.append(evaluated_row(result, run, prediction))
    return base_rows, realtime_rows


def summary_line(name: str, rows: list[dict]) -> str:
    item = summarize(rows)
    count = len(rows)
    return (
        f"| {name} | {count} | {pct(item['outcome_hit'] / count)} | "
        f"{pct(item['top1_bucket_hit'] / count)} | {pct(item['top2_bucket_hit'] / count)} | "
        f"{pct(item['any_exact_hit'] / count)} | {pct(item['any_score_bucket_hit'] / count)} | "
        f"{item['median_deviation']:.3f} |"
    )


def main() -> int:
    base_rows, realtime_rows = comparison_rows()
    if not base_rows or len(base_rows) != len(realtime_rows):
        raise RuntimeError("base and realtime evaluation rows are missing or misaligned")
    context_count = sum(
        prediction.get("context_applied") == "TRUE"
        for _, prediction in load_prediction_archive().values()
    )
    shape_count = sum(
        prediction.get("shape_applied") == "TRUE"
        for _, prediction in load_prediction_archive().values()
    )
    lines = [
        "# Base Model vs Realtime-Assisted System",
        "",
        "Both rows use the same 79 pre-kickoff forecasts and regulation-time results.",
        "The realtime-assisted system used human-collected evidence interpreted with GPT-5.5",
        "at the `very high` reasoning setting.",
        "",
        "| System | Matches | Outcome | Top1 total | Top2 total | Any exact | Any score bucket | Median deviation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        summary_line("Base model", base_rows),
        summary_line("Realtime-assisted", realtime_rows),
        "",
        f"Realtime team context was applied to {context_count}/79 matches; pre-match shape context to {shape_count}/79.",
        "Different search dates, sources, or language models can produce different realtime inputs.",
        "",
        "## Score Columns",
        "",
        "The base model has three historical score columns; the realtime-assisted system has four.",
        "",
        "| System | Score | Available | Exact | Outcome | Bucket | Median deviation |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for system_name, rows in (("Base model", base_rows), ("Realtime-assisted", realtime_rows)):
        for label, _ in SCORE_COLUMNS:
            item = score_column_summary(rows, label)
            if not item["available"]:
                continue
            denominator = item["available"]
            lines.append(
                f"| {system_name} | {label} | {denominator} | "
                f"{pct(item['exact'] / denominator)} | {pct(item['outcome'] / denominator)} | "
                f"{pct(item['bucket'] / denominator)} | {item['median_deviation']:.3f} |"
            )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary: {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
