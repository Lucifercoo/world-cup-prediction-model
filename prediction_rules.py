from __future__ import annotations

import re
from collections.abc import Mapping

TOTAL_GOAL_BUCKET_LABELS = ("0-1球", "2-3球", "4-5球", "6-8球")
_SCORE_PATTERN = re.compile(r"^(\d+)-(\d+)$")


def parse_score(value: str) -> tuple[int, int]:
    match = _SCORE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(f"invalid score: {value}")
    return int(match.group(1)), int(match.group(2))


def format_score(goals_a: int, goals_b: int) -> str:
    return f"{goals_a}-{goals_b}"


def score_outcome(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "A"
    if goals_b > goals_a:
        return "B"
    return "D"


def total_goal_bucket(total_goals: int) -> str:
    if total_goals <= 1:
        return "0-1球"
    if total_goals <= 3:
        return "2-3球"
    if total_goals <= 5:
        return "4-5球"
    return "6-8球"


def match_key(row: Mapping[str, str]) -> tuple[str, str, str, str]:
    return row["date_bjt"], row["time_bjt"], row["team_a"], row["team_b"]


def parse_top2_total_goal_buckets(value: str, primary_bucket: str) -> set[str]:
    buckets = {primary_bucket}
    for part in value.split(";"):
        fields = part.strip().split()
        if not fields:
            continue
        bucket = fields[0]
        if bucket not in TOTAL_GOAL_BUCKET_LABELS:
            raise ValueError(f"unknown total-goal bucket: {bucket}")
        buckets.add(bucket)
        if len(buckets) >= 2:
            break
    return buckets
