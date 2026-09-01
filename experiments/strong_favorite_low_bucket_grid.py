from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate_plan_against_results import evaluate

OUTPUT_DIR = ROOT / "output"
PLAN_CSV = OUTPUT_DIR / "realtime_context_adjusted_plan.csv"
EXPERIMENT_DIR = OUTPUT_DIR / "strong_favorite_low_bucket_grid"
SUMMARY_CSV = EXPERIMENT_DIR / "summary.csv"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def row_key(row: dict) -> tuple[str, str, str, str]:
    return row["date_bjt"], row["time_bjt"], row["team_a"], row["team_b"]


def changed_rows(base_rows: list[dict], variant_rows: list[dict]) -> list[dict]:
    base = {row_key(row): row for row in base_rows}
    changed: list[dict] = []
    for row in variant_rows:
        key = row_key(row)
        old = base.get(key)
        if old is None:
            continue
        if (
            old["adjusted_total_goal_bucket"] != row["adjusted_total_goal_bucket"]
            or old["adjusted_score_1_model"] != row["adjusted_score_1_model"]
        ):
            changed.append(row)
    return changed


def run_variant(label: str, favorite_xg: float, underdog_xg: float, probability: float) -> Path:
    env = dict(os.environ)
    env["WC_STRONG_FAVORITE_LOW_BUCKET_MIN_FAVORITE_XG"] = f"{favorite_xg:.2f}"
    env["WC_STRONG_FAVORITE_LOW_BUCKET_MAX_UNDERDOG_XG"] = f"{underdog_xg:.2f}"
    env["WC_STRONG_FAVORITE_LOW_BUCKET_MIN_PROBABILITY"] = f"{probability:.2f}"
    subprocess.run(
        ["uv", "run", "python", "realtime_context_adjusted_plan.py"],
        cwd=ROOT.parent,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    out = EXPERIMENT_DIR / f"plan_{label}_fxg{favorite_xg:.2f}_uxg{underdog_xg:.2f}_p{probability:.2f}.csv"
    shutil.copyfile(PLAN_CSV, out)
    return out


def main() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    base_path = run_variant("off", 99.00, 0.00, 0.99)
    base_rows = read_csv(base_path)
    base_metrics = evaluate(base_path)

    summary_rows: list[dict] = []
    variants = [
        (2.20, 0.50, 0.58),
        (2.00, 0.50, 0.52),
        (1.80, 0.60, 0.52),
        (1.70, 0.50, 0.45),
        (1.50, 0.50, 0.48),
        (1.50, 0.70, 0.45),
    ]

    for favorite_xg, underdog_xg, probability in variants:
        plan_path = run_variant("grid", favorite_xg, underdog_xg, probability)
        rows = read_csv(plan_path)
        changed = changed_rows(base_rows, rows)
        metrics = evaluate(plan_path)
        summary_rows.append(
            {
                "favorite_xg_min": f"{favorite_xg:.2f}",
                "underdog_xg_max": f"{underdog_xg:.2f}",
                "favorite_probability_min": f"{probability:.2f}",
                "changed_count": str(len(changed)),
                "changed_matches": "; ".join(f"{row['team_a']} vs {row['team_b']}" for row in changed[:12]),
                "outcome_hit": str(metrics["outcome_hit"]),
                "outcome_delta": str(metrics["outcome_hit"] - base_metrics["outcome_hit"]),
                "top1_hit": str(metrics["top1_hit"]),
                "top1_delta": str(metrics["top1_hit"] - base_metrics["top1_hit"]),
                "top2_hit": str(metrics["top2_hit"]),
                "top2_delta": str(metrics["top2_hit"] - base_metrics["top2_hit"]),
                "any_exact": str(metrics["any_exact"]),
                "any_exact_delta": str(metrics["any_exact"] - base_metrics["any_exact"]),
                "any_score_bucket": str(metrics["any_score_bucket"]),
                "any_score_bucket_delta": str(metrics["any_score_bucket"] - base_metrics["any_score_bucket"]),
                "mean_deviation": f"{metrics['mean_deviation']:.6f}",
                "mean_deviation_delta": f"{metrics['mean_deviation'] - base_metrics['mean_deviation']:.6f}",
                "median_deviation": f"{metrics['median_deviation']:.6f}",
                "median_deviation_delta": f"{metrics['median_deviation'] - base_metrics['median_deviation']:.6f}",
                "selected_2_3": str(metrics["selected_2_3"]),
            }
        )

    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    best = sorted(
        summary_rows,
        key=lambda row: (
            -int(row["top1_hit"]),
            -int(row["top2_hit"]),
            float(row["mean_deviation"]),
            -int(row["any_exact"]),
        ),
    )[:10]
    print(f"Summary: {SUMMARY_CSV}")
    for row in best:
        print(row)


if __name__ == "__main__":
    main()
