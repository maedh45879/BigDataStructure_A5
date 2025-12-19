from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from query_simulator.costs import QueryCostModel
from query_simulator.models import CostBreakdown

from .loader import build_database_model, load_queries, load_stats, resolve_schema_root
from .models import PlanQuerySpec
from .planner import build_plans
from .reporting import (
    compare_per_query,
    format_collection_sizes,
    format_leaderboard,
    format_plan,
    format_plan_result,
)
from .simulator import simulate_plan


def _scale_cost(cost: CostBreakdown, factor: float) -> CostBreakdown:
    return CostBreakdown(
        label=f"{cost.label}*{factor}",
        data_scanned_gb=cost.data_scanned_gb * factor,
        time_cost=cost.time_cost * factor,
        carbon_cost=cost.carbon_cost * factor,
        price_cost=cost.price_cost * factor,
        notes=list(cost.notes),
    )


def _add_costs(label: str, costs: List[CostBreakdown]) -> CostBreakdown:
    model = QueryCostModel()
    return model.aggregate(label, costs)


def run(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 5 - Data Model Selection Challenge"
    )
    parser.add_argument(
        "--data-root",
        default=str(Path("data") / "chapter5"),
        help="Root folder for chapter5 datasets",
    )
    parser.add_argument(
        "--schema-set",
        default="default",
        help="Schema set name (default uses data-root directly).",
    )
    parser.add_argument(
        "--dbs",
        nargs="+",
        default=["DB1", "DB2"],
        help="Database signatures to compare.",
    )
    args = parser.parse_args(argv)

    data_root = resolve_schema_root(Path(args.data_root), args.schema_set)
    schema_dir = data_root / "schemas"
    stats_path = data_root / "stats.json"
    queries_path = data_root / "queries.json"

    cluster_config, stats_map = load_stats(stats_path)
    query_specs = [
        PlanQuerySpec(
            name=item["name"],
            description=item.get("description", ""),
            frequency=float(item.get("frequency", 1.0)),
        )
        for item in load_queries(queries_path)
    ]
    plans = build_plans(query_specs)

    totals_by_db: Dict[str, CostBreakdown] = {}
    per_query: Dict[str, Dict[str, CostBreakdown]] = {}

    for db_name in args.dbs:
        db_model = build_database_model(schema_dir, db_name, stats_map)
        print(format_collection_sizes(db_name, db_model))
        print("")

        db_costs: List[CostBreakdown] = []
        for plan in plans:
            print(format_plan(plan))
            result = simulate_plan(plan, db_model, cluster_config)
            print(format_plan_result(result))
            print("")

            weighted = _scale_cost(result.total_cost, plan.query.frequency)
            db_costs.append(weighted)

            per_query.setdefault(plan.query.name, {})[db_name] = result.total_cost

        totals_by_db[db_name] = _add_costs(f"{db_name}:total", db_costs)

    ordering = sorted(
        totals_by_db,
        key=lambda name: (
            totals_by_db[name].price_cost,
            totals_by_db[name].carbon_cost,
            totals_by_db[name].time_cost,
        ),
    )

    print(compare_per_query(per_query, args.dbs))
    print("")
    print(format_leaderboard(totals_by_db, ordering))


if __name__ == "__main__":
    run()
