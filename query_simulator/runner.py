from __future__ import annotations

import argparse
import json
from typing import Iterable, List, Optional

from .config import DEFAULT_DB_SIGNATURE
from .costs import QueryCostModel
from .models import CostBreakdown, QueryCostResult
from .queries import default_queries


def simulate_queries(
    db_signature: str = DEFAULT_DB_SIGNATURE,
    queries: Optional[Iterable] = None,
    model: Optional[QueryCostModel] = None,
) -> List[QueryCostResult]:
    model = model or QueryCostModel()
    queries = list(queries) if queries is not None else default_queries()
    return [model.estimate_query(q, db_signature=db_signature) for q in queries]


def format_cost(cost: CostBreakdown, indent: str = "") -> str:
    notes = f" | {'; '.join(cost.notes)}" if cost.notes else ""
    return (
        f"{indent}{cost.label}: data={cost.data_scanned_gb:.4f} GB, "
        f"time={cost.time_cost:.4f}, carbon={cost.carbon_cost:.4f}, "
        f"price={cost.price_cost:.4f}{notes}"
    )


def format_cost_results(results: Iterable[QueryCostResult]) -> str:
    lines: List[str] = []
    for result in results:
        lines.append(f"{result.query.name} — {result.query.description}")
        lines.append(format_cost(result.total, indent="  "))
        lines.append("  components:")
        for cost in result.component_costs.values():
            lines.append(format_cost(cost, indent="    "))
        if result.join_overhead is not None:
            lines.append("  join overhead:")
            lines.append(format_cost(result.join_overhead, indent="    "))
        lines.append("")
    return "\n".join(lines).rstrip()


def cost_to_dict(cost: CostBreakdown) -> dict:
    return {
        "label": cost.label,
        "data_scanned_gb": cost.data_scanned_gb,
        "time_cost": cost.time_cost,
        "carbon_cost": cost.carbon_cost,
        "price_cost": cost.price_cost,
        "notes": list(cost.notes),
    }


def result_to_dict(result: QueryCostResult) -> dict:
    return {
        "query": {
            "name": result.query.name,
            "kind": result.query.kind,
            "description": result.query.description,
            "join_selectivity": result.query.join_selectivity,
        },
        "total": cost_to_dict(result.total),
        "components": {k: cost_to_dict(v) for k, v in result.component_costs.items()},
        "join_overhead": cost_to_dict(result.join_overhead)
        if result.join_overhead
        else None,
    }


def run_cli(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="QuerySimulator — cost estimation for Q1–Q5."
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_SIGNATURE,
        help=f"Database signature to use (default: {DEFAULT_DB_SIGNATURE})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a formatted text report.",
    )
    args = parser.parse_args(argv)

    results = simulate_queries(db_signature=args.db)

    if args.json:
        serialized = [result_to_dict(r) for r in results]
        print(json.dumps(serialized, indent=2))
    else:
        print(format_cost_results(results))


if __name__ == "__main__":
    run_cli()
