from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

from predict import canonical_team


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
FIFA_RANKING_CSV = DATA_DIR / "fifa_ranking-2022-10-06.csv"
PROFILE_CSV = OUTPUT_DIR / "team_profiles_2022_pre_world_cup.csv"
OUTPUT_CSV = OUTPUT_DIR / "profile_vs_fifa_2022.csv"
OUTPUT_MD = OUTPUT_DIR / "profile_vs_fifa_2022.md"
RANK_DATE = "2022-10-06"


ALIASES = {
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "USA": "United States",
}


def normalized_team(name: str) -> str:
    return canonical_team(ALIASES.get(name.strip(), name.strip()))


def load_profiles() -> list[dict]:
    with PROFILE_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 32:
        raise RuntimeError(f"expected 32 profile rows, got {len(rows)}")
    return rows


def load_fifa_rankings() -> dict[str, dict]:
    rankings: dict[str, dict] = {}
    with FIFA_RANKING_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["rank_date"] != RANK_DATE:
                continue
            team = normalized_team(row["country_full"])
            rankings[team] = {
                "fifa_rank": int(row["rank"]),
                "fifa_points": float(row["total_points"]),
            }
    if len(rankings) < 200:
        raise RuntimeError(f"expected at least 200 FIFA ranking rows for {RANK_DATE}, got {len(rankings)}")
    return rankings


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Pearson correlation needs same-length lists with at least 2 values")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        raise ValueError("Pearson correlation denominator is zero")
    return numerator / (dx * dy)


def rank_values(rows: list[dict], key: str, reverse: bool) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)
    return {row["team"]: index for index, row in enumerate(ordered, start=1)}


def merged_rows() -> list[dict]:
    profiles = load_profiles()
    fifa = load_fifa_rankings()
    missing = sorted(row["team"] for row in profiles if row["team"] not in fifa)
    if missing:
        raise RuntimeError(f"missing FIFA ranking rows: {', '.join(missing)}")

    rows: list[dict] = []
    for row in profiles:
        team = row["team"]
        ranking = fifa[team]
        rows.append(
            {
                "team": team,
                "profile_rank": 0,
                "fifa_rank_among_32": 0,
                "rank_gap_profile_minus_fifa": 0,
                "fifa_rank": ranking["fifa_rank"],
                "fifa_points": ranking["fifa_points"],
                "profile_strength": float(row["strength_score"]),
                "tier": row["tier"],
                "style": row["style"],
            }
        )

    profile_ranks = rank_values(rows, "profile_strength", reverse=True)
    fifa_ranks_among_32 = rank_values(rows, "fifa_points", reverse=True)
    for row in rows:
        row["profile_rank"] = profile_ranks[row["team"]]
        row["fifa_rank_among_32"] = fifa_ranks_among_32[row["team"]]
        row["rank_gap_profile_minus_fifa"] = row["profile_rank"] - row["fifa_rank_among_32"]

    return sorted(rows, key=lambda row: row["profile_rank"])


def write_csv(rows: list[dict]) -> None:
    fields = [
        "team",
        "profile_rank",
        "fifa_rank_among_32",
        "rank_gap_profile_minus_fifa",
        "fifa_rank",
        "fifa_points",
        "profile_strength",
        "tier",
        "style",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict]) -> None:
    corr_rank = pearson(
        [float(row["profile_rank"]) for row in rows],
        [float(row["fifa_rank_among_32"]) for row in rows],
    )
    corr_points = pearson(
        [float(row["profile_strength"]) for row in rows],
        [float(row["fifa_points"]) for row in rows],
    )
    overvalued = sorted(rows, key=lambda row: row["rank_gap_profile_minus_fifa"])[:10]
    undervalued = sorted(rows, key=lambda row: row["rank_gap_profile_minus_fifa"], reverse=True)[:10]

    lines = [
        "# 2022 赛前画像 vs FIFA 排名",
        "",
        f"- FIFA 排名日期：{RANK_DATE}",
        f"- 队伍：2022 世界杯 32 队",
        f"- 画像排名 vs FIFA 队内排名 Pearson：{corr_rank:.3f}",
        f"- 画像强度 vs FIFA 积分 Pearson：{corr_points:.3f}",
        "",
        "## 画像明显高于 FIFA",
        "",
        "| 球队 | 画像排名 | FIFA 32队内排名 | 差值 | FIFA总排名 | 画像强度 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overvalued:
        lines.append(
            "| {team} | {profile_rank} | {fifa_rank_among_32} | {gap} | {fifa_rank} | {strength:.3f} |".format(
                team=row["team"],
                profile_rank=row["profile_rank"],
                fifa_rank_among_32=row["fifa_rank_among_32"],
                gap=row["rank_gap_profile_minus_fifa"],
                fifa_rank=row["fifa_rank"],
                strength=row["profile_strength"],
            )
        )

    lines.extend(
        [
            "",
            "## 画像明显低于 FIFA",
            "",
            "| 球队 | 画像排名 | FIFA 32队内排名 | 差值 | FIFA总排名 | 画像强度 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in undervalued:
        lines.append(
            "| {team} | {profile_rank} | {fifa_rank_among_32} | {gap} | {fifa_rank} | {strength:.3f} |".format(
                team=row["team"],
                profile_rank=row["profile_rank"],
                fifa_rank_among_32=row["fifa_rank_among_32"],
                gap=row["rank_gap_profile_minus_fifa"],
                fifa_rank=row["fifa_rank"],
                strength=row["profile_strength"],
            )
        )

    lines.extend(
        [
            "",
            "## 全量对照",
            "",
            "| 球队 | 画像排名 | FIFA 32队内排名 | FIFA总排名 | FIFA积分 | 画像强度 | 档次 | 风格 |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {team} | {profile_rank} | {fifa_rank_among_32} | {fifa_rank} | {points:.2f} | {strength:.3f} | {tier} | {style} |".format(
                team=row["team"],
                profile_rank=row["profile_rank"],
                fifa_rank_among_32=row["fifa_rank_among_32"],
                fifa_rank=row["fifa_rank"],
                points=row["fifa_points"],
                strength=row["profile_strength"],
                tier=row["tier"],
                style=row["style"],
            )
        )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    rows = merged_rows()
    write_csv(rows)
    write_markdown(rows)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
