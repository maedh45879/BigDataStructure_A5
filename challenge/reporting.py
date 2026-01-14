from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

from .config import WEIGHT_CARBON, WEIGHT_PRICE, WEIGHT_TIME
from .models import QueryResult, QuerySpec, plan_summary


def write_results_csv(
    path: Path,
    rows: Iterable[Dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if not rows_list:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_list[0].keys()))
        writer.writeheader()
        writer.writerows(rows_list)


def leaderboard_md(
    totals: Dict[str, Dict[str, float]],
    ordering: List[str],
) -> str:
    lines = ["# Denormalization Leaderboard", ""]
    lines.append("| Rank | Denorm | Time | Carbon | Price | Weighted |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for rank, denorm_id in enumerate(ordering, start=1):
        total = totals[denorm_id]
        weighted = (
            total["time"] * WEIGHT_TIME
            + total["carbon"] * WEIGHT_CARBON
            + total["price"] * WEIGHT_PRICE
        )
        lines.append(
            f"| {rank} | {denorm_id} | {total['time']:.6f} | {total['carbon']:.6f} | {total['price']:.6f} | {weighted:.6f} |"
        )
    lines.append("")
    lines.append("Weights:")
    lines.append(f"- time: {WEIGHT_TIME}")
    lines.append(f"- carbon: {WEIGHT_CARBON}")
    lines.append(f"- price: {WEIGHT_PRICE}")
    return "\n".join(lines)


def summarize_result(result: QueryResult) -> str:
    return (
        f"{result.plan.query.id} -> time={result.total_cost.time_cost:.6f}, "
        f"carbon={result.total_cost.carbon_cost:.6f}, price={result.total_cost.price_cost:.6f}, "
        f"scanned_docs={result.scanned_docs}, output_docs={result.output_docs}"
    )


def plan_to_json(plan: QuerySpec, results: Dict[str, QueryResult]) -> Dict[str, object]:
    per_denorm = {}
    for denorm_id, result in results.items():
        required_indexes: Dict[str, List[str]] = {}
        for op in result.operators:
            if op.operator.target_collection and op.operator.indexes_used:
                required_indexes.setdefault(op.operator.target_collection, [])
                for index in op.operator.indexes_used:
                    if index not in required_indexes[op.operator.target_collection]:
                        required_indexes[op.operator.target_collection].append(index)
        per_denorm[denorm_id] = {
            "query_id": result.plan.query.id,
            "sql": result.plan.query.sql,
            "involved_collections": result.plan.involved_collections,
            "required_indexes": required_indexes,
            "operators": [
                {
                    "name": op.operator.name,
                    "type": op.operator.operator_type,
                    "target_collection": op.operator.target_collection,
                    "left_collection": op.operator.left_collection,
                    "right_collection": op.operator.right_collection,
                    "filters": [
                        {"field": pred.field, "value": pred.value}
                        for pred in op.operator.filters
                    ],
                    "join": None
                    if op.operator.join is None
                    else {
                        "left": op.operator.join.left_field,
                        "right": op.operator.join.right_field,
                    },
                    "grouping_keys": op.operator.grouping_keys,
                    "output_fields": op.operator.output_fields,
                    "scan_strategy": op.operator.scan_strategy,
                    "indexes_used": op.operator.indexes_used,
                }
                for op in result.operators
            ],
        }
    return {"query_id": plan.id, "per_denorm": per_denorm}


def operator_plan_summary(result: QueryResult) -> str:
    return plan_summary([op.operator for op in result.operators])
