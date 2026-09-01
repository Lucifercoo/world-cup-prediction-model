from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "output" / "finished_realtime_cache_evaluation.csv"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "strict-evaluation.png"
BUCKETS = ("0-1", "2-3", "4-5", "6-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the README evaluation chart from strict cached results."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Evaluation file contains no rows: {path}")
    return rows


def is_true(value: str) -> bool:
    return value.strip().upper() == "TRUE"


def rate(rows: list[dict[str, str]], field: str) -> float:
    return 100.0 * sum(is_true(row[field]) for row in rows) / len(rows)


def bucket_label(value: str) -> str:
    return value.removesuffix("球")


def generate(rows: list[dict[str, str]], output: Path) -> None:
    actual = [
        sum(bucket_label(row["actual_total_bucket"]) == bucket for row in rows)
        for bucket in BUCKETS
    ]
    predicted = [
        sum(bucket_label(row["selected_total_bucket"]) == bucket for row in rows)
        for bucket in BUCKETS
    ]
    metric_labels = (
        "Outcome",
        "Top-1 goal bucket",
        "Top-2 goal buckets",
        "Any exact score",
        "Any score bucket",
    )
    metric_fields = (
        "outcome_hit",
        "top1_bucket_hit",
        "top2_bucket_hit",
        "any_score_exact_hit",
        "any_score_bucket_hit",
    )
    metrics = [rate(rows, field) for field in metric_fields]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#263238",
            "text.color": "#263238",
            "xtick.color": "#455A64",
            "ytick.color": "#455A64",
        }
    )
    fig, (ax_distribution, ax_metrics) = plt.subplots(
        1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": (1, 1.15)}
    )
    fig.patch.set_facecolor("white")

    positions = list(range(len(BUCKETS)))
    width = 0.36
    actual_bars = ax_distribution.bar(
        [position - width / 2 for position in positions],
        actual,
        width,
        label="Actual",
        color="#1565C0",
    )
    predicted_bars = ax_distribution.bar(
        [position + width / 2 for position in positions],
        predicted,
        width,
        label="Top-1 prediction",
        color="#F9A825",
    )
    ax_distribution.set_title("Goal-bucket distribution")
    ax_distribution.set_xlabel("Total goals")
    ax_distribution.set_ylabel("Matches")
    ax_distribution.set_xticks(positions, BUCKETS)
    ax_distribution.set_ylim(0, max(actual + predicted) * 1.25)
    ax_distribution.legend(frameon=False, loc="upper right")
    ax_distribution.bar_label(actual_bars, padding=3)
    ax_distribution.bar_label(predicted_bars, padding=3)

    metric_positions = list(range(len(metric_labels)))
    metric_colors = ["#00897B", "#546E7A", "#00897B", "#546E7A", "#00897B"]
    metric_bars = ax_metrics.barh(
        metric_positions, metrics, color=metric_colors, height=0.62
    )
    ax_metrics.set_title("Strict pre-match evaluation")
    ax_metrics.set_xlabel("Hit rate")
    ax_metrics.set_yticks(metric_positions, metric_labels)
    ax_metrics.set_xlim(0, 100)
    ax_metrics.xaxis.set_major_formatter(PercentFormatter(100))
    ax_metrics.invert_yaxis()
    ax_metrics.bar_label(
        metric_bars,
        labels=[f"{value:.1f}%" for value in metrics],
        padding=5,
        color="#263238",
    )

    for axis in (ax_distribution, ax_metrics):
        axis.set_facecolor("white")
        axis.grid(axis="y" if axis is ax_distribution else "x", alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#CFD8DC")

    fig.suptitle(
        f"2026 World Cup Forecast Evaluation · {len(rows)} Matches",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "Latest prediction cached before kickoff; regulation-time scoring for outcome, totals, and exact scores.",
        ha="center",
        color="#546E7A",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.92), w_pad=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    generate(rows, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
