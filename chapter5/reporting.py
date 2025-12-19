from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from main import DatabaseModel
from query_simulator.models import CostBreakdown

from .models import PlanResult, QueryPlan


def format_collection_sizes(db_name: str, db_model: DatabaseModel) -> str:
    lines = [f"Database {db_name} collections:"]
    for name, collection in db_model.collections.items():
        lines.append(f"  - {name}: {collection.size_gb():.4f} GB")
    return "\n".join(lines)


def format_plan(plan: QueryPlan) -> str:
    lines = [f"Plan for {plan.query.name}: {plan.query.description}"]
    for step in plan.steps:
        if step.operator_type == "filter":
            lines.append(
                f"  - {step.name}: filter {step.target_collection} on {step.filter_key}"
            )
        elif step.operator_type == "join":
            lines.append(
                f"  - {step.name}: join {step.left_ref} x {step.right_ref} on {step.join_key}"
            )
        elif step.operator_type == "aggregate":
            keys = ", ".join(step.grouping_keys)
            lines.append(
                f"  - {step.name}: aggregate {step.target_collection} by [{keys}]"
            )
    return "\n".join(lines)


def format_cost(cost: CostBreakdown, indent: str = "") -> str:
    return (
        f"{indent}{cost.label}: data={cost.data_scanned_gb:.4f} GB, "
        f"time={cost.time_cost:.4f}, carbon={cost.carbon_cost:.4f}, "
        f"price={cost.price_cost:.4f}"
    )


def format_plan_result(result: PlanResult) -> str:
    lines = [
        f"Result for {result.plan.query.name}: docs={result.output_documents}, "
        f"size={result.output_size_gb:.4f} GB"
    ]
    lines.append(format_cost(result.total_cost, indent="  "))
    lines.append("  steps:")
    for step in result.steps:
        lines.append(format_cost(step.cost, indent="    "))
        for detail in step.details:
            lines.append(format_cost(detail, indent="      "))
    return "\n".join(lines)


def format_leaderboard(
    totals: Dict[str, CostBreakdown],
    ordering: List[str],
) -> str:
    lines = ["Leaderboard (lower is better):"]
    for rank, db_name in enumerate(ordering, start=1):
        total = totals[db_name]
        lines.append(
            f"  {rank}. {db_name}: time={total.time_cost:.4f}, "
            f"carbon={total.carbon_cost:.4f}, price={total.price_cost:.4f}"
        )
    return "\n".join(lines)


def compare_per_query(
    per_query: Dict[str, Dict[str, CostBreakdown]],
    db_names: Iterable[str],
) -> str:
    lines: List[str] = ["Per-query costs (time/carbon/price):"]
    for query_name, db_costs in per_query.items():
        lines.append(f"  {query_name}:")
        for db_name in db_names:
            cost = db_costs.get(db_name)
            if cost is None:
                continue
            lines.append(
                f"    - {db_name}: {cost.time_cost:.4f} / "
                f"{cost.carbon_cost:.4f} / {cost.price_cost:.4f}"
            )
    return "\n".join(lines)
