from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
EVAL_CSV = OUTPUT / "finished_realtime_cache_evaluation.csv"
OUT_MD = OUTPUT / "early_knockout_total_cap_experiment.md"

BUCKETS = ("0-1球", "2-3球", "4-5球", "6-8球")
EARLY_KNOCKOUT = {"R32", "R16"}
EXEMPT_LABELS = {"open_mismatch", "collapse_risk"}


@dataclass(frozen=True)
class Variant:
    name: str
    favorite_probability_min: float
    favorite_xg_min: float
    xg_gap_min: float
    underdog_xg_max: float


VARIANTS = (
    Variant("baseline", 1.01, 99.0, 99.0, -1.0),
    Variant("soft_cap", 0.42, 2.05, 0.85, 1.45),
    Variant("control_cap", 0.45, 2.20, 1.00, 1.35),
    Variant("strict_cap", 0.50, 2.35, 1.20, 1.20),
)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_probabilities(raw: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for part in raw.split(";"):
        if not part.strip():
            continue
        label, probability = part.rsplit(" ", 1)
        result[label.strip()] = float(probability.rstrip("%")) / 100.0
    return result


def latest_cache_dir() -> Path:
    cache_root = OUTPUT / "realtime_context_cache"
    return max((p for p in cache_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)


def load_plan(cache_dir: Path) -> dict[tuple[str, str, str], dict]:
    plan_file = cache_dir / "outputs" / "realtime_context_adjusted_plan.csv"
    with plan_file.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_key: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        by_key[(row["date_bjt"], row["team_a"], row["team_b"])] = row
    return by_key


def should_cap(row: dict[str, str], plan_row: dict, variant: Variant) -> bool:
    if variant.name == "baseline":
        return False
    if row["group"] not in EARLY_KNOCKOUT:
        return False
    if row["selected_total_bucket"] not in {"4-5球", "6-8球"}:
        return False

    labels = {label.strip() for label in (plan_row.get("shape_labels") or "").split(";") if label.strip()}
    if labels & EXEMPT_LABELS:
        return False

    home_xg = float(plan_row["adjusted_xg_a"])
    away_xg = float(plan_row["adjusted_xg_b"])
    favorite_xg = max(home_xg, away_xg)
    underdog_xg = min(home_xg, away_xg)
    xg_gap = favorite_xg - underdog_xg
    favorite_probability = max(
        float(plan_row["adjusted_p_a"]),
        float(plan_row["adjusted_p_b"]),
    )

    return (
        favorite_probability >= variant.favorite_probability_min
        and favorite_xg >= variant.favorite_xg_min
        and xg_gap >= variant.xg_gap_min
        and underdog_xg <= variant.underdog_xg_max
    )


def selected_bucket(row: dict[str, str], plan_row: dict, variant: Variant) -> str:
    if should_cap(row, plan_row, variant):
        return "2-3球"
    return row["selected_total_bucket"]


def selected_top2(row: dict[str, str], plan_row: dict, variant: Variant) -> set[str]:
    probabilities = parse_probabilities(row["raw_total_goal_buckets"])
    if should_cap(row, plan_row, variant):
        source = max(probabilities.get("4-5球", 0.0), probabilities.get("6-8球", 0.0))
        probabilities["2-3球"] = max(probabilities.get("2-3球", 0.0), source + 0.001)
        probabilities["4-5球"] = max(probabilities.get("4-5球", 0.0), source * 0.45)
        probabilities["6-8球"] = min(probabilities.get("6-8球", 0.0), probabilities["4-5球"] * 0.35)
    ranked = sorted(BUCKETS, key=lambda b: probabilities.get(b, 0.0), reverse=True)
    return set(ranked[:2])


def evaluate() -> str:
    cache_dir = latest_cache_dir()
    plan = load_plan(cache_dir)
    with EVAL_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    lines = [
        "# Early Knockout Total Cap Experiment",
        "",
        f"Cache: `{cache_dir.name}`",
        "",
        "| variant | top1 bucket | top2 bucket | R32 top1 | R32 top2 | changed | fixed top1 | broken top1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    baseline_top1 = {
        (r["date_bjt"], r["team_a"], r["team_b"]): parse_bool(r["top1_bucket_hit"])
        for r in rows
    }

    details: list[str] = []
    for variant in VARIANTS:
        top1 = top2 = r32_top1 = r32_top2 = changed = fixed = broken = 0
        r32_count = 0
        changed_rows: list[str] = []
        for row in rows:
            key = (row["date_bjt"], row["team_a"], row["team_b"])
            plan_row = plan[key]
            bucket = selected_bucket(row, plan_row, variant)
            top2_set = selected_top2(row, plan_row, variant)
            actual = row["actual_total_bucket"]
            hit1 = bucket == actual
            hit2 = actual in top2_set
            top1 += int(hit1)
            top2 += int(hit2)
            if row["group"] in EARLY_KNOCKOUT:
                r32_count += 1
                r32_top1 += int(hit1)
                r32_top2 += int(hit2)
            if bucket != row["selected_total_bucket"]:
                changed += 1
                was = baseline_top1[key]
                fixed += int(hit1 and not was)
                broken += int((not hit1) and was)
                changed_rows.append(
                    f"- {row['team_a']} vs {row['team_b']}: "
                    f"{row['selected_total_bucket']} -> {bucket}, actual {actual}"
                )

        total = len(rows)
        lines.append(
            f"| {variant.name} | {top1}/{total} {top1 / total:.1%} | "
            f"{top2}/{total} {top2 / total:.1%} | "
            f"{r32_top1}/{r32_count} {r32_top1 / r32_count:.1%} | "
            f"{r32_top2}/{r32_count} {r32_top2 / r32_count:.1%} | "
            f"{changed} | {fixed} | {broken} |"
        )
        if changed_rows:
            details.append(f"## {variant.name}\n\n" + "\n".join(changed_rows))

    lines.extend(["", *details, ""])
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


if __name__ == "__main__":
    print(evaluate())
