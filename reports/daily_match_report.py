from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
PLAN_CSV = OUTPUT_DIR / "realtime_context_adjusted_plan.csv"
LIVE_RANKING_CSV = OUTPUT_DIR / "world_cup_2026_live_rankings.csv"
REPORT_DIR = OUTPUT_DIR / "daily_reports"

TEAM_ZH = {
    "Algeria": "阿尔及利亚",
    "Argentina": "阿根廷",
    "Australia": "澳大利亚",
    "Austria": "奥地利",
    "Belgium": "比利时",
    "Bosnia and Herzegovina": "波黑",
    "Brazil": "巴西",
    "Canada": "加拿大",
    "Cape Verde": "佛得角",
    "Colombia": "哥伦比亚",
    "Croatia": "克罗地亚",
    "Curaçao": "库拉索",
    "Czechia": "捷克",
    "DR Congo": "刚果（金）",
    "Ecuador": "厄瓜多尔",
    "Egypt": "埃及",
    "England": "英格兰",
    "France": "法国",
    "Germany": "德国",
    "Ghana": "加纳",
    "Haiti": "海地",
    "Iran": "伊朗",
    "Iraq": "伊拉克",
    "Ivory Coast": "科特迪瓦",
    "Japan": "日本",
    "Jordan": "约旦",
    "Mexico": "墨西哥",
    "Morocco": "摩洛哥",
    "Netherlands": "荷兰",
    "New Zealand": "新西兰",
    "Norway": "挪威",
    "Panama": "巴拿马",
    "Paraguay": "巴拉圭",
    "Portugal": "葡萄牙",
    "Qatar": "卡塔尔",
    "Saudi Arabia": "沙特阿拉伯",
    "Scotland": "苏格兰",
    "Senegal": "塞内加尔",
    "South Africa": "南非",
    "South Korea": "韩国",
    "Spain": "西班牙",
    "Sweden": "瑞典",
    "Switzerland": "瑞士",
    "Tunisia": "突尼斯",
    "Turkey": "土耳其",
    "United States": "美国",
    "Uruguay": "乌拉圭",
    "Uzbekistan": "乌兹别克斯坦",
}


def zh_team(name: str) -> str:
    return TEAM_ZH.get(name, name)


def zh_text(text: str) -> str:
    out = text or ""
    for en in sorted(TEAM_ZH, key=len, reverse=True):
        out = out.replace(en, TEAM_ZH[en])
    out = out.replace("FIFA", "FIFA")
    return out


def row_outcome_zh(row: dict[str, str]) -> str:
    outcome = row["predicted_outcome"]
    if outcome == "A":
        return f"{zh_team(row['team_a'])}胜"
    if outcome == "B":
        return f"{zh_team(row['team_b'])}胜"
    return "平局"


def team_shape_summary(reason: str) -> str:
    if not reason:
        return ""
    text = zh_text(reason)
    text = re.sub(r", 权重[0-9.]+", "", text)
    text = text.replace("controlled_favorite;low_conversion;low_event", "热门控场但低转化")
    text = text.replace("low_conversion;low_event", "低转化/低事件")
    text = text.replace("collapse_risk;open_game;transition_dog", "开放对攻/防线波动")
    text = text.replace("open_game;transition_dog", "开放比赛/反击路线")
    text = text.replace("controlled_favorite;transition_dog", "热门控场/反击路线")
    text = text.replace("controlled_mismatch", "强弱悬殊控场")
    text = text.replace("controlled_favorite", "热门控场")
    text = text.replace("transition_dog", "弱队反击")
    text = text.replace("collapse_risk", "崩盘风险")
    text = text.replace("open_favorite", "热门进攻")
    text = text.replace("open_game", "开放比赛")
    text = text.replace(";", "/")
    parts = [part.strip() for part in text.split("|") if part.strip()]
    return "；".join(parts)


def situation_summary(row: dict[str, str]) -> str:
    return zh_text(row["group_context_notes"]).replace("；", "；")


def read_live_rankings() -> dict[str, dict[str, str]]:
    with LIVE_RANKING_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        return {row["team"]: row for row in csv.DictReader(fh)}


def signed_float(value: str) -> str:
    number = float(value)
    return f"{number:+.1f}"


def signed_int(value: str) -> str:
    number = int(value)
    return f"{number:+d}"


def ranking_item(team: str, rankings: dict[str, dict[str, str]]) -> str:
    row = rankings[team]
    return (
        f"{zh_team(team)}：实时{row['live_rank']}，"
        f"FIFA{row['base_fifa_rank']}，"
        f"{signed_float(row['points_delta'])}分，"
        f"进失{row['goals_for']}-{row['goals_against']}，"
        f"净胜{signed_int(row['goal_diff'])}"
    )


def ranking_summary(row: dict[str, str], rankings: dict[str, dict[str, str]]) -> str:
    return "；".join(
        [
            ranking_item(row["team_a"], rankings),
            ranking_item(row["team_b"], rankings),
        ]
    )


def previous_summary(row: dict[str, str]) -> str:
    a = team_shape_summary(row["team_shape_reason_a"])
    b = team_shape_summary(row["team_shape_reason_b"])
    return f"{zh_team(row['team_a'])}：{a}；{zh_team(row['team_b'])}：{b}"


def compact_risk(row: dict[str, str]) -> str:
    label = row["risk_label"]
    reasons = zh_text(row["risk_reasons"])
    if reasons:
        return f"{label}（{reasons}）"
    return label


def read_rows(match_date: str) -> list[dict[str, str]]:
    with PLAN_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if row["date_bjt"] == match_date]
    rows.sort(key=lambda row: row["time_bjt"])
    return rows


def run_prediction() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "realtime_context_adjusted_plan.py")],
        cwd=ROOT,
        check=True,
    )


def render_markdown(
    match_date: str,
    rows: list[dict[str, str]],
    rankings: dict[str, dict[str, str]],
) -> str:
    lines = [
        f"# {match_date} 世界杯预测",
        "",
        "| 时间 | 比赛 | 实时 ranking | 此前本届表现 | 胜负参考 | 风险 | 总球 | 模型 / 备选 / 身价 / 爆冷 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        match = f"{zh_team(row['team_a'])} vs {zh_team(row['team_b'])}"
        total_goals = f"主{row['adjusted_total_goal_bucket']}，备{row['backup_total_goal_bucket']}"
        scores = " / ".join(
            [
                row["adjusted_score_1_model"],
                row["adjusted_score_2_aggressive_prediction"],
                row["adjusted_score_3_market_value"],
                row["adjusted_score_4_upset"],
            ]
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    row["time_bjt"],
                    match,
                    ranking_summary(row, rankings),
                    previous_summary(row),
                    row_outcome_zh(row),
                    compact_risk(row),
                    total_goals,
                    scores,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "说明：胜负只做参考；四个比分列固定为“模型 / 备选 / 身价 / 爆冷”。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily World Cup prediction report.")
    parser.add_argument(
        "match_date",
        nargs="?",
        default=date.today().isoformat(),
        help="Match date in Beijing time, e.g. 2026-06-28.",
    )
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Do not rerun realtime_context_adjusted_plan.py before generating report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.no_refresh:
        run_prediction()
    rows = read_rows(args.match_date)
    if not rows:
        raise SystemExit(f"No matches found for {args.match_date} in {PLAN_CSV}")
    rankings = read_live_rankings()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"{args.match_date}.md"
    out_path.write_text(render_markdown(args.match_date, rows, rankings), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
