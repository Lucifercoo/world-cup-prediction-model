from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REQUIRED_MODEL_INPUTS = (
    "fifa_rankings_annual_start.csv",
    "transfermarkt_world_cup_2026_values.csv",
    "world_cup_2026_key_player_signals.csv",
)
OPTIONAL_EMPTY_MODEL_INPUTS = {"world_cup_2026_key_player_signals.csv"}


@dataclass(frozen=True)
class PipelineStep:
    label: str
    module: str


@dataclass(frozen=True)
class Experiment:
    module: str
    description: str


PIPELINE_STEPS = (
    PipelineStep("rolling team profiles", "profiles"),
    PipelineStep("live tournament rankings", "builders.build_live_world_cup_rankings"),
    PipelineStep("in-tournament adjustments", "builders.build_in_tournament_adjustments"),
    PipelineStep("in-tournament shape profiles", "builders.build_in_tournament_shape_profiles"),
    PipelineStep("historical style matchup edges", "builders.build_style_matchup_edges"),
    PipelineStep("base predictions", "predict_fifa_profile"),
    PipelineStep("realtime predictions and cache", "realtime_context_adjusted_plan"),
    PipelineStep("betting-plan output", "betting_plan_fifa_profile"),
)

EXPERIMENTS = {
    "early-knockout-total-cap": Experiment(
        "experiments.early_knockout_total_cap_experiment",
        "Test high-goal caps in R32 and R16 matches.",
    ),
    "joint-total-score": Experiment(
        "experiments.joint_total_score_experiment",
        "Evaluate total-goal buckets and score coverage jointly.",
    ),
    "knockout-draw-score": Experiment(
        "experiments.knockout_draw_score_experiment",
        "Test draw-oriented knockout score allocation.",
    ),
    "knockout-strong-favorite-total": Experiment(
        "experiments.knockout_strong_favorite_total_experiment",
        "Test knockout total-goal controls for strong favorites.",
    ),
    "low-block-effect": Experiment(
        "experiments.low_block_effect_experiment",
        "Measure the aggregate effect of low-block labels.",
    ),
    "low-block-mechanism": Experiment(
        "experiments.low_block_mechanism_experiment",
        "Compare low-block adjustment mechanisms.",
    ),
    "outcome-conflict-score": Experiment(
        "experiments.outcome_conflict_score_experiment",
        "Test score allocation when outcome signals conflict.",
    ),
    "strong-favorite-low-bucket-grid": Experiment(
        "experiments.strong_favorite_low_bucket_grid",
        "Grid-search strong-favorite protection thresholds.",
    ),
    "total-goal-bucket-score": Experiment(
        "experiments.total_goal_bucket_score_experiment",
        "Compare score selection within total-goal buckets.",
    ),
    "xg-model-legacy-backup": Experiment(
        "experiments.xg_model_legacy_backup_experiment",
        "Compare xG model scores with the legacy backup path.",
    ),
    "xg-score-generation": Experiment(
        "experiments.xg_score_generation_experiment",
        "Evaluate xG-based score generation variants.",
    ),
}


def run_module(module: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT,
        check=True,
    )


def require_model_inputs() -> None:
    problems: list[str] = []
    for name in REQUIRED_MODEL_INPUTS:
        path = DATA_DIR / name
        schema_path = ROOT / "schemas" / name
        if not path.is_file():
            problems.append(f"data/{name} is missing (schema: schemas/{name})")
            continue
        with schema_path.open(encoding="utf-8-sig", newline="") as schema_handle:
            expected_fields = csv.DictReader(schema_handle).fieldnames
        with path.open(encoding="utf-8-sig", newline="") as data_handle:
            reader = csv.DictReader(data_handle)
            if reader.fieldnames != expected_fields:
                problems.append(f"data/{name} does not match schemas/{name}")
            elif next(reader, None) is None and name not in OPTIONAL_EMPTY_MODEL_INPUTS:
                problems.append(f"data/{name} contains no data rows")
    if not problems:
        return
    raise RuntimeError("invalid required model inputs: " + "; ".join(problems))


def run_pipeline(*, replay: bool) -> None:
    require_model_inputs()
    total = len(PIPELINE_STEPS)
    for index, step in enumerate(PIPELINE_STEPS, start=1):
        print(f"[{index}/{total}] {step.label}", flush=True)
        arguments = ("--replay",) if replay and step.module.endswith("build_in_tournament_adjustments") else ()
        run_module(step.module, *arguments)


def run_data(args: argparse.Namespace) -> None:
    arguments: list[str] = []
    if args.list:
        arguments.append("--list")
    if args.build:
        arguments.append("--build")
    if args.overwrite:
        arguments.append("--overwrite")
    for name in args.file:
        arguments.extend(("--file", name))
    run_module("scripts.fetch_data", *arguments)


def run_setup(args: argparse.Namespace) -> None:
    run_module("scripts.fetch_data")
    run_module("scripts.prepare_public_data")
    run_module("builders.fetch_wikipedia_squad_club_cohesion")
    if not args.data_only:
        run_pipeline(replay=False)


def run_report(args: argparse.Namespace) -> None:
    if args.no_build and args.replay:
        raise RuntimeError("--replay cannot be used with --no-build")
    if not args.no_build:
        run_pipeline(replay=args.replay)
    run_module("reports.daily_match_report", args.match_date, "--no-refresh")


def run_context(args: argparse.Namespace) -> None:
    if args.context_command == "prepare":
        run_module(
            "scripts.prepare_realtime_context_package",
            args.input,
            "--output-dir",
            args.output_dir,
        )
    else:
        arguments = [args.package_dir]
        if args.data_dir:
            arguments.extend(("--data-dir", args.data_dir))
        run_module("scripts.apply_realtime_context_package", *arguments)


def list_experiments() -> None:
    for name, experiment in EXPERIMENTS.items():
        print(f"{name:<38} {experiment.description}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the World Cup prediction project.")
    commands = parser.add_subparsers(dest="command", required=True)

    data_parser = commands.add_parser("data", help="Fetch or build reviewed datasets.")
    data_mode = data_parser.add_mutually_exclusive_group()
    data_mode.add_argument("--list", action="store_true", help="List acquisition modes.")
    data_mode.add_argument("--build", action="store_true", help="Run build-mode data commands.")
    data_parser.add_argument("--overwrite", action="store_true", help="Replace mismatched downloads.")
    data_parser.add_argument("--file", action="append", default=[], help="Limit work to a named file.")

    build_command = commands.add_parser("build", help="Run the full prediction pipeline.")
    build_command.add_argument(
        "--replay",
        action="store_true",
        help="Rebuild the in-tournament event ledger from all recorded results.",
    )

    setup_parser = commands.add_parser(
        "setup",
        help="Prepare public inputs and run the complete pipeline.",
    )
    setup_parser.add_argument(
        "--data-only",
        action="store_true",
        help="Prepare public inputs without running predictions.",
    )

    report_parser = commands.add_parser("report", help="Generate a dated Markdown report.")
    report_parser.add_argument("match_date", nargs="?", default=date.today().isoformat())
    report_parser.add_argument("--no-build", action="store_true", help="Use existing prediction outputs.")
    report_parser.add_argument("--replay", action="store_true", help="Replay state before reporting.")

    evaluate_parser = commands.add_parser("evaluate", help="Evaluate strict pre-match predictions.")
    evaluate_parser.add_argument(
        "--source",
        choices=("archive", "cache"),
        default="archive",
        help="Use the public compact archive or a local full cache.",
    )
    evaluate_parser.add_argument("--cache-dir", help="Full cache path used with --source cache.")

    inspect_parser = commands.add_parser("inspect", help="Inspect generated match predictions.")
    inspect_parser.add_argument("--date", help="Filter by Beijing date (YYYY-MM-DD).")
    inspect_parser.add_argument("--team", help="Filter by English team name.")
    inspect_parser.add_argument("--input", help="Use a different prediction CSV.")

    context_parser = commands.add_parser("context", help="Prepare or apply realtime context.")
    context_commands = context_parser.add_subparsers(dest="context_command", required=True)
    context_prepare = context_commands.add_parser("prepare", help="Validate LLM context JSON.")
    context_prepare.add_argument("input")
    context_prepare.add_argument("--output-dir", default="output/context-package")
    context_apply = context_commands.add_parser("apply", help="Apply a validated context package.")
    context_apply.add_argument("package_dir")
    context_apply.add_argument("--data-dir")

    experiment_parser = commands.add_parser("experiment", help="List or run isolated experiments.")
    experiment_commands = experiment_parser.add_subparsers(dest="experiment_command", required=True)
    experiment_commands.add_parser("list", help="List registered experiments.")
    experiment_run = experiment_commands.add_parser("run", help="Run one registered experiment.")
    experiment_run.add_argument("name", choices=tuple(EXPERIMENTS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "data":
        run_data(args)
    elif args.command == "setup":
        run_setup(args)
    elif args.command == "build":
        run_pipeline(replay=args.replay)
    elif args.command == "report":
        run_report(args)
    elif args.command == "evaluate":
        arguments = ["--source", args.source]
        if args.cache_dir:
            arguments.extend(("--cache-dir", args.cache_dir))
        run_module("evaluation.evaluate_finished_from_realtime_cache", *arguments)
        if args.source == "archive":
            run_module("evaluation.compare_base_and_realtime")
    elif args.command == "inspect":
        arguments = []
        if args.date:
            arguments.extend(("--date", args.date))
        if args.team:
            arguments.extend(("--team", args.team))
        if args.input:
            arguments.extend(("--input", args.input))
        run_module("scripts.inspect_predictions", *arguments)
    elif args.command == "context":
        run_context(args)
    elif args.experiment_command == "list":
        list_experiments()
    else:
        run_module(EXPERIMENTS[args.name].module)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
