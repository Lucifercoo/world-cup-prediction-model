from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

BJT = timezone(timedelta(hours=8))
REALTIME_OUTPUT_FIELDS = (
    "date_bjt",
    "time_bjt",
    "group",
    "team_a",
    "team_b",
    "venue",
    "predicted_outcome",
    "p_a",
    "p_draw",
    "p_b",
    "adjusted_p_a",
    "adjusted_p_draw",
    "adjusted_p_b",
    "xg_a",
    "xg_b",
    "adjusted_xg_a",
    "adjusted_xg_b",
    "selected_total_goal_bucket",
    "adjusted_total_goal_bucket",
    "backup_total_goal_bucket",
    "adjusted_total_goals_top2",
    "bucket_primary_score",
    "adjusted_score_1_model",
    "bucket_complement_score",
    "aggressive_score",
    "adjusted_score_2_aggressive_prediction",
    "market_value_raw_score",
    "market_value_score",
    "adjusted_score_3_market_value",
    "upset_score",
    "upset_score_probability",
    "adjusted_score_4_upset",
    "adjusted_score_4_upset_probability",
    "xg_goal_diff",
    "xg_outcome_edge",
    "legacy_outcome_edge",
    "outcome_edge_conflict",
    "risk_label",
    "risk_reasons",
    "context_applied",
    "shape_applied",
    "shape_labels",
    "shape_notes",
    "style_features_a",
    "style_features_b",
    "style_matchup_edge",
    "style_matchup_influence",
    "style_matchup_points_shift",
    "style_matchup_total_multiplier",
    "style_matchup_reasons",
    "team_shape_labels_a",
    "team_shape_labels_b",
    "team_shape_reason_a",
    "team_shape_reason_b",
    "team_shape_profile_mode",
    "group_round",
    "draw_acceptance_a",
    "draw_acceptance_b",
    "group_draw_multiplier",
    "group_tempo_multiplier",
    "group_context_complete",
    "group_context_notes",
    "context_chain_multipliers",
    "context_signal_tags",
    "source_confidence_a",
    "source_confidence_b",
    "lineup_certainty_a",
    "lineup_certainty_b",
    "context_notes",
    "context_sources",
)


def outcome_label(row: dict) -> str:
    if row["predicted_outcome"] == "A":
        return f"{row['team_a']}胜"
    if row["predicted_outcome"] == "B":
        return f"{row['team_b']}胜"
    if row["predicted_outcome"] == "D":
        return "平局"
    raise ValueError(f"unknown outcome: {row['predicted_outcome']}")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REALTIME_OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REALTIME_OUTPUT_FIELDS})


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# 实时战队情况调整版方案",
        "",
        "- 调整 xG、平局概率、精确总进球数和比分项。",
        "- 比赛形态层可改变胜负参考。",
        "- 小组形势层按轮次和赛前积分调整平局接受度。",
        "- 缺失字段不猜，按 1.00 处理。",
        "",
        "| 北京时间 | 比赛 | 胜负参考 | 风险提示 | 调整概率 | 小组形势 | 形态 | 实时链路 | 原总进球 | 总进球Top2 | 模型 | 备选 | 身价 | 爆冷 | xG变化 |",
        "|---|---|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["context_applied"] != "TRUE":
            continue
        lines.append(
            "| {date} {time} | {match} | {outcome} | {risk} | {probs} | {group_context} | {shape} | {chain} | {old_bucket} | {new_bucket} | "
            "{model} | {aggressive} | {market} | {upset} | {old_xg}->{new_xg} |".format(
                date=row["date_bjt"],
                time=row["time_bjt"],
                match=f"{row['team_a']} vs {row['team_b']}",
                outcome=outcome_label(row),
                risk=row.get("risk_label", ""),
                probs=(
                    f"{float(row.get('adjusted_p_a') or row['p_a']):.1%}/"
                    f"{float(row.get('adjusted_p_draw') or row['p_draw']):.1%}/"
                    f"{float(row.get('adjusted_p_b') or row['p_b']):.1%}"
                ),
                group_context=row.get("group_context_notes", ""),
                shape=row.get("shape_labels", ""),
                chain=row.get("context_signal_tags", ""),
                old_bucket=row["selected_total_goal_bucket"],
                new_bucket=row.get("adjusted_total_goals_top2", row["adjusted_total_goal_bucket"]),
                model=row["adjusted_score_1_model"],
                aggressive=row["adjusted_score_2_aggressive_prediction"],
                market=row["adjusted_score_3_market_value"],
                upset=row.get("adjusted_score_4_upset") or row.get("upset_score", ""),
                old_xg=f"{float(row['xg_a']):.2f}-{float(row['xg_b']):.2f}",
                new_xg=f"{float(row['adjusted_xg_a']):.2f}-{float(row['adjusted_xg_b']):.2f}",
            )
        )
    lines.extend(["", "## 实时依据", ""])
    for row in rows:
        if row["context_applied"] != "TRUE":
            continue
        lines.extend(
            [
                f"### {row['team_a']} vs {row['team_b']}",
                "",
                f"形态：{row.get('shape_labels', '')}",
                "",
                f"小组形势：{row.get('group_context_notes', '')}",
                "",
                row.get("shape_notes", ""),
                "",
                row["context_notes"],
                "",
                row["context_sources"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_file(source: Path, target_dir: Path, name: str) -> dict:
    target_dir.mkdir(parents=True, exist_ok=True)
    item = {
        "name": name,
        "source": str(source),
        "exists": source.exists(),
        "sha256": file_sha256(source),
    }
    if source.exists():
        target = target_dir / source.name
        shutil.copy2(source, target)
        item["cached_path"] = str(target)
    return item


def write_realtime_cache(
    rows: list[dict],
    *,
    cache_dir: Path,
    source_files: list[tuple[str, Path]],
    output_files: list[tuple[str, Path]],
    runtime_parameters: dict[str, object],
) -> str:
    run_id = datetime.now(BJT).strftime("%Y%m%d_%H%M%S_%f")
    run_dir = cache_dir / run_id
    manifest = {
        "run_id": run_id,
        "created_at_bjt": datetime.now(BJT).isoformat(timespec="seconds"),
        "script": "realtime_context_adjusted_plan.py",
        "team_shape_profile_mode": runtime_parameters.get("team_shape_profile_mode", ""),
        "runtime_parameters": runtime_parameters,
        "row_count": len(rows),
        "context_applied_count": sum(1 for row in rows if row["context_applied"] == "TRUE"),
        "shape_applied_count": sum(1 for row in rows if row["shape_applied"] == "TRUE"),
        "input_files": [cache_file(path, run_dir / "inputs", name) for name, path in source_files],
        "output_files": [cache_file(path, run_dir / "outputs", name) for name, path in output_files],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (run_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    (cache_dir / "latest_run_id.txt").write_text(run_id + "\n", encoding="utf-8")
    (cache_dir / "latest_manifest.json").write_text(manifest_text, encoding="utf-8")
    return str(run_dir)
