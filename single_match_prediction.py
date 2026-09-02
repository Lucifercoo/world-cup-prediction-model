from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import predict_fifa_profile as model
import realtime_context_adjusted_plan as realtime
from predict import Match, canonical_team
from prediction_rules import format_score, parse_score
from reports.daily_match_report import TEAM_ZH

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output" / "single_match_predictions"
PROFILE_CUTOFF = date(2026, 6, 12)
LIVE_RESULTS_CSV = DATA_DIR / "world_cup_2026_results.csv"
MARKET_VALUE_CSV = DATA_DIR / "transfermarkt_world_cup_2026_values.csv"
COHESION_CSV = DATA_DIR / "world_cup_2026_team_club_cohesion.csv"
NEUTRAL_MARKET_VALUE = model.MarketValue(total_eur_m=149.0, average_eur_m=149.0 / 26)
NEUTRAL_COHESION = model.ClubCohesion(
    top_club="",
    top_club_players=0,
    max_club_share=0.0,
    top3_club_share=0.0,
    multiplier=1.0,
)
STAGES = {
    "friendly": "FRIENDLY",
    "qualifier": "QUALIFIER",
    "group": "GROUP",
    "r32": "R32",
    "r16": "R16",
    "qf": "QF",
    "sf": "SF",
    "final": "FINAL",
    "third-place": "3P",
}

ZH_TO_EN = {zh: en for en, zh in TEAM_ZH.items()}
ZH_TO_EN.update({"美国队": "United States", "韩国队": "South Korea"})


@dataclass(frozen=True)
class DataStatus:
    item: str
    used: bool
    status: str
    as_of: str | None
    source: str


def parse_kickoff(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("kickoff must use ISO 8601, including timezone") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("kickoff must include a timezone, for example +08:00")
    return parsed


def resolve_team(value: str) -> str:
    return canonical_team(ZH_TO_EN.get(value.strip(), value.strip()))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def latest_result_date() -> date | None:
    if not LIVE_RESULTS_CSV.is_file():
        return None
    dates = [date.fromisoformat(row["date_bjt"]) for row in read_csv(LIVE_RESULTS_CSV)]
    return max(dates) if dates else None


def optional_market_values(teams: tuple[str, str]) -> tuple[dict[str, model.MarketValue], bool, str]:
    if not MARKET_VALUE_CSV.is_file():
        return {team: NEUTRAL_MARKET_VALUE for team in teams}, False, "文件不存在"
    rows = read_csv(MARKET_VALUE_CSV)
    by_team = {canonical_team(row["team"]): row for row in rows}
    missing = [team for team in teams if team not in by_team]
    if missing:
        return {team: NEUTRAL_MARKET_VALUE for team in teams}, False, f"缺少球队：{', '.join(missing)}"
    values = {
        team: model.MarketValue(
            total_eur_m=float(by_team[team]["market_value_eur_m"]),
            average_eur_m=float(by_team[team]["average_market_value_eur_m"]),
        )
        for team in teams
    }
    sources = {by_team[team]["source"] for team in teams}
    source_label = "公开代理数据" if any("PROXY" in source for source in sources) else "用户提供数据"
    return values, True, source_label


def optional_cohesion(teams: tuple[str, str]) -> tuple[dict[str, model.ClubCohesion], bool, str]:
    if not COHESION_CSV.is_file():
        return {team: NEUTRAL_COHESION for team in teams}, False, "文件不存在"
    rows = read_csv(COHESION_CSV)
    by_team = {canonical_team(row["team"]): row for row in rows}
    missing = [team for team in teams if team not in by_team]
    if missing:
        return {team: NEUTRAL_COHESION for team in teams}, False, f"缺少球队：{', '.join(missing)}"
    values = {
        team: model.ClubCohesion(
            top_club=by_team[team]["top_club"],
            top_club_players=int(by_team[team]["top_club_players"]),
            max_club_share=float(by_team[team]["max_club_share"]),
            top3_club_share=float(by_team[team]["top3_club_share"]),
            multiplier=float(by_team[team]["club_cohesion_multiplier"]),
        )
        for team in teams
    }
    return values, True, "2026 世界杯名单"


@contextmanager
def without_tournament_adjustments():
    original = model.IN_TOURNAMENT_ADJUSTMENTS
    model.IN_TOURNAMENT_ADJUSTMENTS = {}
    try:
        yield
    finally:
        model.IN_TOURNAMENT_ADJUSTMENTS = original


def internal_match(
    team_a: str,
    team_b: str,
    kickoff: datetime,
    stage: str,
    venue: str,
    home: str,
) -> Match:
    kickoff_bjt = kickoff.astimezone(timezone(timedelta(hours=8)))
    internal_time = kickoff_bjt - timedelta(hours=12)
    home_team = "" if home == "neutral" else team_a if home == "a" else team_b
    return Match(
        STAGES[stage],
        internal_time.date(),
        internal_time.strftime("%H:%M"),
        team_a,
        team_b,
        venue,
        home_team,
    )


def outcome_text(code: str, team_a_zh: str, team_b_zh: str) -> str:
    return {"A": f"{team_a_zh}胜", "D": "平局", "B": f"{team_b_zh}胜"}[code]


def stale_status(as_of: date, kickoff_date: date) -> str:
    age = (kickoff_date - as_of).days
    if age < 0:
        raise RuntimeError(f"data dated {as_of} is later than kickoff {kickoff_date}")
    return f"未自动更新，距开赛 {age} 天"


def apply_stage_rules(raw: dict, stage: str, market_values: dict[str, model.MarketValue]) -> dict:
    if stage in {"friendly", "qualifier", "group"}:
        return raw
    return realtime.apply_context(
        raw,
        contexts={},
        shapes={},
        team_shape_profiles={},
        key_player_signals={},
        key_player_statuses={},
        team_market_values={team: value.total_eur_m for team, value in market_values.items()},
        completed_matches=[],
    )


def remove_unavailable_market_influence(prediction: dict) -> dict:
    selected_bucket = prediction.get("adjusted_total_goal_bucket", prediction["selected_total_goal_bucket"])
    top2 = prediction.get("adjusted_total_goals_top2", prediction["top_total_goal_buckets"])
    buckets = {selected_bucket}
    buckets.update(part.split()[0] for part in top2.split(";") if part.strip())
    p_a = float(prediction.get("adjusted_p_a", prediction["p_a"]))
    p_draw = float(prediction.get("adjusted_p_draw", prediction["p_draw"]))
    p_b = float(prediction.get("adjusted_p_b", prediction["p_b"]))
    lambda_a = float(prediction.get("adjusted_xg_a", prediction["xg_a"]))
    lambda_b = float(prediction.get("adjusted_xg_b", prediction["xg_b"]))
    cells = model.outcome_adjusted_scores(lambda_a, lambda_b, p_a, p_draw, p_b)
    model_score = prediction.get("adjusted_score_1_model", prediction["bucket_primary_score"])
    backup_score = prediction.get("adjusted_score_2_aggressive_prediction", prediction["bucket_complement_score"])
    upset = model.select_upset_or_compression_score(
        cells,
        buckets,
        p_a,
        p_draw,
        p_b,
        {parse_score(model_score), parse_score(backup_score)},
    )
    return {
        **prediction,
        "adjusted_score_4_upset": format_score(upset[0], upset[1]),
    }


def build_prediction(args: argparse.Namespace) -> dict:
    team_a = resolve_team(args.team_a)
    team_b = resolve_team(args.team_b)
    if team_a == team_b:
        raise RuntimeError("team-a and team-b must be different")
    kickoff_bjt = args.kickoff.astimezone(timezone(timedelta(hours=8)))
    teams = (team_a, team_b)

    if not model.PROFILE_CSV.is_file() or (
        not model.LIVE_RANKING_CSV.is_file() and not model.FIFA_RANKING_CSV.is_file()
    ):
        raise RuntimeError("缺少核心排名或球队画像，请先运行 `uv run python -m wc_model setup`")
    rankings = model.load_fifa_rankings()
    profiles = model.load_profiles()
    for team in teams:
        if team not in rankings:
            raise RuntimeError(f"缺少 {team} 的 FIFA 排名")
        if team not in profiles:
            raise RuntimeError(f"缺少 {team} 的十年球队画像；当前版本只覆盖 48 支世界杯球队")

    using_live_ranking = model.LIVE_RANKING_CSV.is_file()
    result_as_of = latest_result_date() if using_live_ranking else None
    if result_as_of and result_as_of > kickoff_bjt.date():
        raise RuntimeError(
            f"本地实时排名包含开赛后的赛果（更新至 {result_as_of}），不能用于该场预测"
        )

    market_values, market_used, market_note = optional_market_values(teams)
    cohesion, cohesion_used, cohesion_note = optional_cohesion(teams)
    baselines = model.profile_baselines(list(profiles.values()))
    match = internal_match(team_a, team_b, args.kickoff, args.stage, args.venue, args.home)
    with without_tournament_adjustments():
        raw = model.predict_match(match, rankings, profiles, baselines, market_values, cohesion)
    selected = apply_stage_rules(raw, args.stage, market_values)
    if not market_used:
        selected = remove_unavailable_market_influence(selected)

    team_a_zh = TEAM_ZH.get(team_a, team_a)
    team_b_zh = TEAM_ZH.get(team_b, team_b)
    ranking_snapshot = rankings[team_a].snapshot_date.removeprefix("live:")
    ranking_as_of = result_as_of or date.fromisoformat(ranking_snapshot)
    data_status = [
        DataStatus(
            "FIFA/赛事排名",
            True,
            stale_status(ranking_as_of, kickoff_bjt.date()),
            ranking_as_of.isoformat(),
            (
                f"FIFA 快照 {ranking_snapshot}"
                + (f" + 世界杯赛果更新至 {result_as_of}" if result_as_of else "")
            ),
        ),
        DataStatus(
            "十年球队画像",
            True,
            stale_status(PROFILE_CUTOFF, kickoff_bjt.date()),
            PROFILE_CUTOFF.isoformat(),
            "过去十年国际比赛，三年半衰期",
        ),
        DataStatus("阵容身价", market_used, market_note, "2026 世界杯周期" if market_used else None, market_note),
        DataStatus("俱乐部集中度", cohesion_used, cohesion_note, "2026 世界杯名单" if cohesion_used else None, cohesion_note),
        DataStatus("实时首发/伤停/天气/关键球员", False, "用户未提供，本次未使用", None, "无"),
        DataStatus("世界杯赛中状态", False, "任意单场默认不使用，避免跨赛事误用", None, "无"),
        DataStatus(
            "赛事阶段规则",
            args.stage not in {"friendly", "qualifier", "group"},
            (
                "缺少小组成员、轮次和积分，本次不使用小组形势修正"
                if args.stage == "group"
                else "已应用淘汰赛通用规则"
                if args.stage not in {"friendly", "qualifier"}
                else "友谊赛/预选赛不应用世界杯阶段规则"
            ),
            None,
            args.stage,
        ),
    ]
    mode = "增强预测" if market_used and cohesion_used else "核心预测"
    scores = {
        "模型": selected.get("adjusted_score_1_model", selected["bucket_primary_score"]),
        "备选": selected.get("adjusted_score_2_aggressive_prediction", selected["bucket_complement_score"]),
        "身价": selected.get("adjusted_score_3_market_value", selected["market_value_score"]) if market_used else "不可用",
        "爆冷": selected.get("adjusted_score_4_upset", selected["upset_score"]),
    }
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "match": {
            "team_a": team_a,
            "team_a_zh": team_a_zh,
            "team_b": team_b,
            "team_b_zh": team_b_zh,
            "kickoff": args.kickoff.isoformat(),
            "kickoff_bjt": kickoff_bjt.isoformat(),
            "stage": args.stage,
            "venue": args.venue,
            "home": args.home,
        },
        "prediction": {
            "outcome": outcome_text(selected["predicted_outcome"], team_a_zh, team_b_zh),
            "probabilities": {
                team_a_zh: float(selected.get("adjusted_p_a", selected["p_a"])),
                "平局": float(selected.get("adjusted_p_draw", selected["p_draw"])),
                team_b_zh: float(selected.get("adjusted_p_b", selected["p_b"])),
            },
            "xg": {
                team_a_zh: float(selected.get("adjusted_xg_a", selected["xg_a"])),
                team_b_zh: float(selected.get("adjusted_xg_b", selected["xg_b"])),
            },
            "total_goals_top1": selected.get("adjusted_total_goal_bucket", selected["selected_total_goal_bucket"]),
            "total_goals_top2": next(
                part.split()[0]
                for part in selected.get("adjusted_total_goals_top2", selected["top_total_goal_buckets"]).split(";")
                if part.strip()
                and part.split()[0]
                != selected.get("adjusted_total_goal_bucket", selected["selected_total_goal_bucket"])
            ),
            "scores": scores,
            "risk": selected["risk_label"],
            "risk_reasons": selected["risk_reasons"],
        },
        "teams": {
            team_a_zh: {"rank": rankings[team_a].rank, "points": rankings[team_a].points, "style": profiles[team_a].style},
            team_b_zh: {"rank": rankings[team_b].rank, "points": rankings[team_b].points, "style": profiles[team_b].style},
        },
        "data_status": [asdict(item) for item in data_status],
    }


def render_markdown(result: dict) -> str:
    match = result["match"]
    prediction = result["prediction"]
    probabilities = " / ".join(f"{name} {value:.1%}" for name, value in prediction["probabilities"].items())
    xg = " / ".join(f"{name} {value:.2f}" for name, value in prediction["xg"].items())
    scores = " / ".join(f"{name} {value}" for name, value in prediction["scores"].items())
    home_labels = {
        "a": f"{match['team_a_zh']}主场",
        "b": f"{match['team_b_zh']}主场",
        "neutral": "中立场",
    }
    lines = [
        f"# {match['team_a_zh']} vs {match['team_b_zh']}",
        "",
        f"- 预测模式：{result['mode']}",
        f"- 开赛时间：{match['kickoff_bjt']}（北京时间）",
        f"- 场地：{match['venue']}；{home_labels[match['home']]}",
        "",
        "| 项目 | 结果 |",
        "|---|---|",
        f"| 胜负参考 | {prediction['outcome']} |",
        f"| 胜平负概率 | {probabilities} |",
        f"| 期望进球 | {xg} |",
        f"| 总球 | Top-1 {prediction['total_goals_top1']} / Top-2 {prediction['total_goals_top2']} |",
        f"| 比分 | {scores} |",
        f"| 风险 | {prediction['risk']}：{prediction['risk_reasons']} |",
        "",
        "## 数据状态",
        "",
        "| 数据 | 是否使用 | 截止时间 | 状态 | 来源 |",
        "|---|---|---|---|---|",
    ]
    for item in result["data_status"]:
        lines.append(
            f"| {item['item']} | {'是' if item['used'] else '否'} | {item['as_of'] or '-'} | {item['status']} | {item['source']} |"
        )
    return "\n".join(lines) + "\n"


def output_stem(result: dict) -> str:
    match = result["match"]
    slug = re.sub(r"[^a-z0-9]+", "-", f"{match['team_a']}-{match['team_b']}".lower()).strip("-")
    kickoff_date = match["kickoff_bjt"][:10].replace("-", "")
    return f"{kickoff_date}_{slug}"


def write_outputs(result: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(result)
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    return json_path, markdown_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one match using existing local data.")
    parser.add_argument("--team-a", required=True, help="English or Chinese team name.")
    parser.add_argument("--team-b", required=True, help="English or Chinese team name.")
    parser.add_argument("--kickoff", required=True, type=parse_kickoff, help="ISO 8601 time with timezone.")
    parser.add_argument("--stage", required=True, choices=tuple(STAGES))
    parser.add_argument("--venue", default="未指定")
    parser.add_argument("--home", choices=("a", "b", "neutral"), default="neutral")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    args = parse_args()
    result = build_prediction(args)
    json_path, markdown_path = write_outputs(result, args.output_dir)
    print(render_markdown(result))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
