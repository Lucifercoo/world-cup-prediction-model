from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_CSV = ROOT / "output" / "finished_realtime_cache_evaluation.csv"
CACHE_DIR = ROOT / "output" / "realtime_context_cache"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SUMMARY_MD = OUTPUT_DIR / "knockout_draw_score_experiment_summary.md"
DETAIL_CSV = OUTPUT_DIR / "knockout_draw_score_experiment_details.csv"

SCORE_LABELS = ("model", "aggressive", "market", "upset")
KNOCKOUT_GROUPS = {"R32", "R16", "QF", "SF", "FINAL", "3P"}
LOW_EVENT_LABELS = {"low_block", "low_event", "low_event_favorite", "controlled_favorite", "credible_opponent"}
LOW_EVENT_DRAW_LABELS = {"low_block", "low_event", "low_event_favorite"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["date_bjt"], row["time_bjt"], row["team_a"], row["team_b"]


def parse_score(value: str) -> tuple[int, int]:
    a, b = value.split("-", maxsplit=1)
    return int(a), int(b)


def format_score(score: tuple[int, int]) -> str:
    return f"{score[0]}-{score[1]}"


def outcome(score: tuple[int, int]) -> str:
    if score[0] > score[1]:
        return "A"
    if score[1] > score[0]:
        return "B"
    return "D"


def total_bucket(total: int) -> str:
    if total <= 1:
        return "0-1球"
    if total <= 3:
        return "2-3球"
    if total <= 5:
        return "4-5球"
    return "6-8球"


def bucket_values(bucket: str) -> set[int]:
    if bucket == "0-1球":
        return {0, 1}
    if bucket == "2-3球":
        return {2, 3}
    if bucket == "4-5球":
        return {4, 5}
    if bucket == "6-8球":
        return {6, 7, 8}
    return set()


def deviation(actual: str, predicted: str) -> float:
    actual_a, actual_b = parse_score(actual)
    pred_a, pred_b = parse_score(predicted)
    return (abs(actual_a - pred_a) + abs(actual_b - pred_b)) / max(1, actual_a + actual_b)


def score_for_outcome(bucket: str, predicted_outcome: str, favorite: str, p_draw: float) -> str:
    values = bucket_values(bucket)
    if predicted_outcome == "D":
        if 2 in values:
            return "1-1"
        if 0 in values:
            return "0-0"
        if 4 in values:
            return "2-2"
        return "3-3"

    if bucket == "0-1球":
        return "1-0" if predicted_outcome == "A" else "0-1"

    if bucket == "2-3球":
        if p_draw >= 0.29 and 2 in values:
            return "1-1"
        return "2-1" if predicted_outcome == "A" else "1-2"

    if bucket == "4-5球":
        return "3-1" if predicted_outcome == "A" else "1-3"

    if favorite == predicted_outcome:
        return "4-2" if predicted_outcome == "A" else "2-4"
    return "3-3"


def load_plan_cache(row: dict[str, str]) -> dict[str, str]:
    plan_path = CACHE_DIR / row["cache_run_id"] / "outputs" / "realtime_context_adjusted_plan.csv"
    rows = {row_key(item): item for item in read_csv(plan_path)}
    return rows[row_key(row)]


def top_non_draw(row: dict[str, str]) -> tuple[str, float]:
    p_a = float(row["adjusted_p_a"])
    p_b = float(row["adjusted_p_b"])
    return ("A", p_a) if p_a >= p_b else ("B", p_b)


def draw_candidate(row: dict[str, str], margin: float, low_event_margin: float) -> bool:
    if row["group"] not in KNOCKOUT_GROUPS:
        return False
    plan = row["_plan"]
    p_draw = float(plan["adjusted_p_draw"])
    _, top = top_non_draw(plan)
    labels = {label for label in plan.get("shape_labels", "").split(";") if label}
    allowed_margin = low_event_margin if labels & LOW_EVENT_LABELS else margin
    return top - p_draw <= allowed_margin


def refined_draw_candidate(row: dict[str, str]) -> bool:
    if row["group"] not in KNOCKOUT_GROUPS:
        return False
    if row["selected_total_bucket"] == "0-1球":
        return False
    plan = row["_plan"]
    p_draw = float(plan["adjusted_p_draw"])
    _, top = top_non_draw(plan)
    edge = top - p_draw
    if edge <= 0.10:
        return True
    labels = {label for label in plan.get("shape_labels", "").split(";") if label}
    return bool(labels & LOW_EVENT_DRAW_LABELS) and p_draw >= 0.28 and edge <= 0.31


def weak_goal_candidate(row: dict[str, str], xg_min: float) -> bool:
    if row["group"] not in KNOCKOUT_GROUPS:
        return False
    plan = row["_plan"]
    score = parse_score(row["model_score"])
    if sum(score) != 2 or min(score) != 0:
        return False
    if row["selected_total_bucket"] != "2-3球":
        return False
    predicted = row["predicted_outcome"]
    if predicted == "A":
        return float(plan["adjusted_xg_b"]) >= xg_min
    if predicted == "B":
        return float(plan["adjusted_xg_a"]) >= xg_min
    return False


def apply_variant(row: dict[str, str], variant: str) -> dict[str, str]:
    changed = dict(row)
    plan = row["_plan"]
    favorite, _ = top_non_draw(plan)
    p_draw = float(plan["adjusted_p_draw"])
    bucket = row["selected_total_bucket"]

    if variant == "draw_close":
        if draw_candidate(row, margin=0.08, low_event_margin=0.18):
            changed["predicted_outcome"] = "D"
            changed["model_score"] = score_for_outcome(bucket, "D", favorite, p_draw)

    elif variant == "draw_wide_low_event":
        if draw_candidate(row, margin=0.10, low_event_margin=0.31):
            changed["predicted_outcome"] = "D"
            changed["model_score"] = score_for_outcome(bucket, "D", favorite, p_draw)

    elif variant == "draw_wide_plus_weak_goal":
        if draw_candidate(row, margin=0.10, low_event_margin=0.31):
            changed["predicted_outcome"] = "D"
            changed["model_score"] = score_for_outcome(bucket, "D", favorite, p_draw)
        elif weak_goal_candidate(row, xg_min=0.45):
            changed["model_score"] = score_for_outcome(bucket, row["predicted_outcome"], favorite, p_draw)

    elif variant == "draw_refined":
        if refined_draw_candidate(row):
            changed["predicted_outcome"] = "D"
            changed["model_score"] = score_for_outcome(bucket, "D", favorite, p_draw)

    changed["changed"] = str(
        changed["predicted_outcome"] != row["predicted_outcome"] or changed["model_score"] != row["model_score"]
    ).upper()
    changed["change_note"] = ""
    if changed["changed"] == "TRUE":
        changed["change_note"] = f"{row['predicted_outcome']} {row['model_score']} -> {changed['predicted_outcome']} {changed['model_score']}"
    return changed


def metric(rows: list[dict[str, str]]) -> dict[str, float | int]:
    deviations: list[float] = []
    model_deviations: list[float] = []
    knockout_rows = [row for row in rows if row["group"] in KNOCKOUT_GROUPS]
    knockout_draw_predictions = sum(row["predicted_outcome"] == "D" for row in knockout_rows)
    knockout_draw_actual = sum(row["actual_outcome"] == "D" for row in knockout_rows)
    any_exact = 0
    any_bucket = 0
    any_outcome = 0
    outcome_hit = 0
    model_exact = 0
    model_bucket = 0
    for row in rows:
        actual = row["actual_score"]
        actual_outcome = row["actual_outcome"]
        actual_bucket = row["actual_total_bucket"]
        scores = [row.get(f"{label}_score", "") for label in SCORE_LABELS if row.get(f"{label}_score", "")]
        row_deviations = [deviation(actual, score) for score in scores]
        deviations.append(sum(row_deviations) / len(row_deviations))
        model_deviation = deviation(actual, row["model_score"])
        model_deviations.append(model_deviation)
        parsed_scores = [parse_score(score) for score in scores]
        outcome_hit += row["predicted_outcome"] == actual_outcome
        model_exact += row["model_score"] == actual
        model_bucket += total_bucket(sum(parse_score(row["model_score"]))) == actual_bucket
        any_exact += any(score == actual for score in scores)
        any_bucket += any(total_bucket(sum(score)) == actual_bucket for score in parsed_scores)
        any_outcome += any(outcome(score) == actual_outcome for score in parsed_scores)
    return {
        "matches": len(rows),
        "outcome_hit": int(outcome_hit),
        "model_exact": int(model_exact),
        "model_bucket": int(model_bucket),
        "any_exact": int(any_exact),
        "any_bucket": int(any_bucket),
        "any_outcome": int(any_outcome),
        "mean_deviation": sum(deviations) / len(deviations),
        "median_deviation": statistics.median(deviations),
        "model_mean_deviation": sum(model_deviations) / len(model_deviations),
        "knockout_matches": len(knockout_rows),
        "knockout_draw_predictions": knockout_draw_predictions,
        "knockout_draw_actual": knockout_draw_actual,
        "changed": sum(row.get("changed") == "TRUE" for row in rows),
    }


def pct(count: float, total: float) -> str:
    if total <= 0:
        return "0.0%"
    return f"{count / total:.1%}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_rows = read_csv(EVAL_CSV)
    for row in baseline_rows:
        row["_plan"] = load_plan_cache(row)
        row["changed"] = "FALSE"
        row["change_note"] = ""

    variants = {
        "baseline": baseline_rows,
        "draw_close": [apply_variant(row, "draw_close") for row in baseline_rows],
        "draw_wide_low_event": [apply_variant(row, "draw_wide_low_event") for row in baseline_rows],
        "draw_wide_plus_weak_goal": [apply_variant(row, "draw_wide_plus_weak_goal") for row in baseline_rows],
        "draw_refined": [apply_variant(row, "draw_refined") for row in baseline_rows],
    }
    metrics = {name: metric(rows) for name, rows in variants.items()}

    detail_rows = []
    best_rows = variants["draw_refined"]
    for base, changed in zip(baseline_rows, best_rows, strict=True):
        if changed.get("changed") == "TRUE":
            detail_rows.append(
                {
                    "date_bjt": base["date_bjt"],
                    "time_bjt": base["time_bjt"],
                    "group": base["group"],
                    "team_a": base["team_a"],
                    "team_b": base["team_b"],
                    "actual_score": base["actual_score"],
                    "actual_outcome": base["actual_outcome"],
                    "actual_total_bucket": base["actual_total_bucket"],
                    "selected_total_bucket": base["selected_total_bucket"],
                    "before": f"{base['predicted_outcome']} {base['model_score']}",
                    "after": f"{changed['predicted_outcome']} {changed['model_score']}",
                    "adjusted_p_a": base["_plan"]["adjusted_p_a"],
                    "adjusted_p_draw": base["_plan"]["adjusted_p_draw"],
                    "adjusted_p_b": base["_plan"]["adjusted_p_b"],
                    "adjusted_xg_a": base["_plan"]["adjusted_xg_a"],
                    "adjusted_xg_b": base["_plan"]["adjusted_xg_b"],
                    "shape_labels": base["_plan"].get("shape_labels", ""),
                }
            )
    fields = [
        "date_bjt",
        "time_bjt",
        "group",
        "team_a",
        "team_b",
        "actual_score",
        "actual_outcome",
        "actual_total_bucket",
        "selected_total_bucket",
        "before",
        "after",
        "adjusted_p_a",
        "adjusted_p_draw",
        "adjusted_p_b",
        "adjusted_xg_a",
        "adjusted_xg_b",
        "shape_labels",
    ]
    write_csv(DETAIL_CSV, detail_rows, fields)

    lines = [
        "# Knockout Draw Score Experiment",
        "",
        "Strict pre-match cache evaluation. Only model score/outcome is altered in variants; total-goal buckets and the other three score columns are unchanged.",
        "",
        "| Variant | Outcome | Model exact | Model bucket | Any exact | Any bucket | Any outcome | Mean dev | Median dev | Model mean dev | KO draw pred/actual | Changed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in metrics.items():
        matches = int(stats["matches"])
        lines.append(
            f"| {name} | {stats['outcome_hit']}/{matches} {pct(stats['outcome_hit'], matches)} | "
            f"{stats['model_exact']}/{matches} {pct(stats['model_exact'], matches)} | "
            f"{stats['model_bucket']}/{matches} {pct(stats['model_bucket'], matches)} | "
            f"{stats['any_exact']}/{matches} {pct(stats['any_exact'], matches)} | "
            f"{stats['any_bucket']}/{matches} {pct(stats['any_bucket'], matches)} | "
            f"{stats['any_outcome']}/{matches} {pct(stats['any_outcome'], matches)} | "
            f"{stats['mean_deviation']:.3f} | {stats['median_deviation']:.3f} | "
            f"{stats['model_mean_deviation']:.3f} | "
            f"{stats['knockout_draw_predictions']}/{stats['knockout_draw_actual']} | "
            f"{stats['changed']} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY_MD)
    print(DETAIL_CSV)
    for name, stats in metrics.items():
        print(name, stats)


if __name__ == "__main__":
    main()
