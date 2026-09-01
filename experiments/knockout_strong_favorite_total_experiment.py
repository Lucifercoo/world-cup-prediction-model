from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_CSV = ROOT / "output" / "finished_realtime_cache_evaluation.csv"
CACHE_DIR = ROOT / "output" / "realtime_context_cache"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
SUMMARY_MD = OUTPUT_DIR / "knockout_strong_favorite_total_experiment_summary.md"
DETAIL_CSV = OUTPUT_DIR / "knockout_strong_favorite_total_experiment_details.csv"

SCORE_LABELS = ("model", "aggressive", "market", "upset")
KNOCKOUT_GROUPS = {"R32", "R16", "QF", "SF", "FINAL", "3P"}
OPEN_LABELS = {"open_game", "open_mismatch", "collapse_risk"}
LOW_EVENT_HIGH_BUCKET_LABELS = {"low_block", "low_event", "low_event_favorite", "controlled_favorite"}
EARLY_HIGH_BUCKET_CAP_STAGES = {"R32", "R16"}
LATE_HIGH_BUCKET_CAP_STAGES = {"QF", "SF", "FINAL"}


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


def total_bucket(total: int) -> str:
    if total <= 1:
        return "0-1球"
    if total <= 3:
        return "2-3球"
    if total <= 5:
        return "4-5球"
    return "6-8球"


def outcome(score: tuple[int, int]) -> str:
    if score[0] > score[1]:
        return "A"
    if score[1] > score[0]:
        return "B"
    return "D"


def deviation(actual: str, predicted: str) -> float:
    actual_a, actual_b = parse_score(actual)
    pred_a, pred_b = parse_score(predicted)
    return (abs(actual_a - pred_a) + abs(actual_b - pred_b)) / max(1, actual_a + actual_b)


def format_score(score: tuple[int, int]) -> str:
    return f"{score[0]}-{score[1]}"


def score_for_bucket(bucket: str, predicted_outcome: str) -> str:
    if predicted_outcome == "D":
        if bucket == "0-1球":
            return "0-0"
        if bucket == "2-3球":
            return "1-1"
        if bucket == "4-5球":
            return "2-2"
        return "3-3"
    if bucket == "0-1球":
        return "1-0" if predicted_outcome == "A" else "0-1"
    if bucket == "2-3球":
        return "2-1" if predicted_outcome == "A" else "1-2"
    if bucket == "4-5球":
        return "3-1" if predicted_outcome == "A" else "1-3"
    return "4-2" if predicted_outcome == "A" else "2-4"


def load_plan_cache(row: dict[str, str]) -> dict[str, str]:
    plan_path = CACHE_DIR / row["cache_run_id"] / "outputs" / "realtime_context_adjusted_plan.csv"
    rows = {row_key(item): item for item in read_csv(plan_path)}
    return rows[row_key(row)]


def candidate(row: dict[str, str], favorite_min: float, xg_min: float, xg_gap_min: float) -> bool:
    if row["group"] not in KNOCKOUT_GROUPS:
        return False
    if row["selected_total_bucket"] != "6-8球":
        return False
    if row["predicted_outcome"] == "D":
        return False
    plan = row["_plan"]
    p_a = float(plan["adjusted_p_a"])
    p_b = float(plan["adjusted_p_b"])
    favorite_probability = max(p_a, p_b)
    favorite_xg = max(float(plan["adjusted_xg_a"]), float(plan["adjusted_xg_b"]))
    xg_gap = abs(float(plan["adjusted_xg_a"]) - float(plan["adjusted_xg_b"]))
    labels = {label for label in plan.get("shape_labels", "").split(";") if label}
    stage = row["group"].upper()
    if stage not in EARLY_HIGH_BUCKET_CAP_STAGES and stage not in LATE_HIGH_BUCKET_CAP_STAGES:
        return False
    if stage in EARLY_HIGH_BUCKET_CAP_STAGES and labels & OPEN_LABELS:
        return False
    if stage in LATE_HIGH_BUCKET_CAP_STAGES and labels & OPEN_LABELS and not labels & LOW_EVENT_HIGH_BUCKET_LABELS:
        return False
    return favorite_probability >= favorite_min and favorite_xg >= xg_min and xg_gap >= xg_gap_min


def apply_variant(row: dict[str, str], variant: str) -> dict[str, str]:
    changed = dict(row)
    if variant == "baseline":
        changed["changed"] = "FALSE"
        return changed

    params = {
        "cap_4_5": (0.66, 3.00, 2.50, "4-5球", "2-3球"),
        "cap_4_5_wide": (0.62, 2.60, 1.80, "4-5球", "2-3球"),
        "low_event_2_3": (0.66, 3.00, 2.50, "shape", "shape"),
    }[variant]
    favorite_min, xg_min, xg_gap_min, primary_bucket, backup_bucket = params
    if not candidate(row, favorite_min, xg_min, xg_gap_min):
        changed["changed"] = "FALSE"
        return changed

    if primary_bucket == "shape":
        labels = {label for label in row["_plan"].get("shape_labels", "").split(";") if label}
        p_draw = float(row["_plan"]["adjusted_p_draw"])
        primary_bucket = "2-3球" if labels & LOW_EVENT_HIGH_BUCKET_LABELS and p_draw >= 0.28 else "4-5球"
        backup_bucket = "4-5球" if primary_bucket == "2-3球" else "2-3球"

    changed["selected_total_bucket"] = primary_bucket
    changed["model_score"] = score_for_bucket(primary_bucket, row["predicted_outcome"])
    changed["aggressive_score"] = score_for_bucket(backup_bucket, row["predicted_outcome"])
    changed["market_score"] = score_for_bucket(primary_bucket, row["predicted_outcome"])
    changed["changed"] = "TRUE"
    changed["change_note"] = (
        f"{row['selected_total_bucket']} {row['model_score']}/{row['aggressive_score']}/{row['market_score']} "
        f"-> {primary_bucket} {changed['model_score']}/{changed['aggressive_score']}/{changed['market_score']}"
    )
    return changed


def metric(rows: list[dict[str, str]]) -> dict[str, float | int]:
    deviations: list[float] = []
    outcome_hit = 0
    top1 = 0
    top2 = 0
    any_exact = 0
    any_bucket = 0
    any_outcome = 0
    changed_count = 0
    for row in rows:
        actual = row["actual_score"]
        actual_score = parse_score(actual)
        actual_bucket = row["actual_total_bucket"]
        top2_buckets = set(row["selected_top2_buckets"].split(","))
        if row.get("changed") == "TRUE":
            changed_count += 1
            top2_buckets = {row["selected_total_bucket"], "4-5球" if row["selected_total_bucket"] == "2-3球" else "2-3球"}
        scores = [row.get(f"{label}_score", "") for label in SCORE_LABELS if row.get(f"{label}_score", "")]
        parsed_scores = [parse_score(score) for score in scores]
        row_deviations = [deviation(actual, score) for score in scores]
        deviations.append(sum(row_deviations) / len(row_deviations))
        outcome_hit += row["predicted_outcome"] == row["actual_outcome"]
        top1 += row["selected_total_bucket"] == actual_bucket
        top2 += actual_bucket in top2_buckets
        any_exact += any(score == actual for score in scores)
        any_bucket += any(total_bucket(sum(score)) == actual_bucket for score in parsed_scores)
        any_outcome += any(outcome(score) == outcome(actual_score) for score in parsed_scores)
    return {
        "matches": len(rows),
        "outcome_hit": int(outcome_hit),
        "top1": int(top1),
        "top2": int(top2),
        "any_exact": int(any_exact),
        "any_bucket": int(any_bucket),
        "any_outcome": int(any_outcome),
        "mean_deviation": sum(deviations) / len(deviations),
        "median_deviation": statistics.median(deviations),
        "changed": changed_count,
    }


def pct(count: float, total: float) -> str:
    return f"{count / total:.1%}" if total else "0.0%"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_rows = read_csv(EVAL_CSV)
    for row in baseline_rows:
        row["_plan"] = load_plan_cache(row)
        row["changed"] = "FALSE"

    variants = {
        name: [apply_variant(row, name) for row in baseline_rows]
        for name in ("baseline", "cap_4_5", "cap_4_5_wide", "low_event_2_3")
    }
    metrics = {name: metric(rows) for name, rows in variants.items()}

    detail_rows: list[dict[str, str]] = []
    for base, changed in zip(baseline_rows, variants["low_event_2_3"], strict=True):
        if changed.get("changed") != "TRUE":
            continue
        detail_rows.append(
            {
                "date_bjt": base["date_bjt"],
                "time_bjt": base["time_bjt"],
                "group": base["group"],
                "team_a": base["team_a"],
                "team_b": base["team_b"],
                "actual_score": base["actual_score"],
                "actual_total_bucket": base["actual_total_bucket"],
                "before_bucket": base["selected_total_bucket"],
                "after_bucket": changed["selected_total_bucket"],
                "before_scores": " / ".join(base[f"{label}_score"] for label in SCORE_LABELS),
                "after_scores": " / ".join(changed[f"{label}_score"] for label in SCORE_LABELS),
                "adjusted_p_a": base["_plan"]["adjusted_p_a"],
                "adjusted_p_draw": base["_plan"]["adjusted_p_draw"],
                "adjusted_p_b": base["_plan"]["adjusted_p_b"],
                "adjusted_xg_a": base["_plan"]["adjusted_xg_a"],
                "adjusted_xg_b": base["_plan"]["adjusted_xg_b"],
                "shape_labels": base["_plan"].get("shape_labels", ""),
                "change_note": changed.get("change_note", ""),
            }
        )
    fields = [
        "date_bjt",
        "time_bjt",
        "group",
        "team_a",
        "team_b",
        "actual_score",
        "actual_total_bucket",
        "before_bucket",
        "after_bucket",
        "before_scores",
        "after_scores",
        "adjusted_p_a",
        "adjusted_p_draw",
        "adjusted_p_b",
        "adjusted_xg_a",
        "adjusted_xg_b",
        "shape_labels",
        "change_note",
    ]
    write_csv(DETAIL_CSV, detail_rows, fields)

    lines = [
        "# Knockout Strong Favorite Total Experiment",
        "",
        "Strict pre-match cache evaluation. Variants only alter knockout matches with primary `6-8球` and a clear non-draw favorite.",
        "",
        "| Variant | Outcome | Top1 | Top2 | Any exact | Any bucket | Any outcome | Mean dev | Median dev | Changed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in metrics.items():
        matches = int(stats["matches"])
        lines.append(
            f"| {name} | {stats['outcome_hit']}/{matches} {pct(stats['outcome_hit'], matches)} | "
            f"{stats['top1']}/{matches} {pct(stats['top1'], matches)} | "
            f"{stats['top2']}/{matches} {pct(stats['top2'], matches)} | "
            f"{stats['any_exact']}/{matches} {pct(stats['any_exact'], matches)} | "
            f"{stats['any_bucket']}/{matches} {pct(stats['any_bucket'], matches)} | "
            f"{stats['any_outcome']}/{matches} {pct(stats['any_outcome'], matches)} | "
            f"{stats['mean_deviation']:.3f} | {stats['median_deviation']:.3f} | {stats['changed']} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY_MD)
    print(DETAIL_CSV)
    for name, stats in metrics.items():
        print(name, stats)


if __name__ == "__main__":
    main()
