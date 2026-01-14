from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .loader import (
    build_database_models,
    collect_embed_paths,
    load_denormalizations,
    load_queries,
    load_schema,
    load_stats,
)
from .models import QuerySpec
from .planner import plan_query
from .reporting import leaderboard_md, operator_plan_summary, plan_to_json, summarize_result, write_results_csv
from .simulator import simulate_plan


def _load_query_specs(queries_path: Path, defaults: Dict[str, float]) -> List[QuerySpec]:
    specs: List[QuerySpec] = []
    for raw in load_queries(queries_path):
        freq = float(raw.get("frequency", defaults.get(raw["id"], 1.0)))
        specs.append(QuerySpec(id=raw["id"], sql=raw["sql"], frequency=freq))
    return specs


def run(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chapter 5 - Data Model Selection Challenge")
    parser.add_argument("--schema", type=str, default="challenge/schema.json")
    parser.add_argument("--stats", type=str, default="challenge/stats.json")
    parser.add_argument("--denorm", type=str, default="challenge/denormalizations.json")
    parser.add_argument("--queries", type=str, default="challenge/queries.json")
    parser.add_argument("--out", type=str, default="out")
    args = parser.parse_args(argv)

    schema_path = Path(args.schema)
    stats_path = Path(args.stats)
    denorm_path = Path(args.denorm)
    queries_path = Path(args.queries)
    out_dir = Path(args.out)
    plans_dir = out_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    schemas = load_schema(schema_path)
    cluster, stats, freq_defaults = load_stats(stats_path)
    denorms = load_denormalizations(denorm_path)
    queries = _load_query_specs(queries_path, freq_defaults)

    per_query_plan_json: Dict[str, Dict[str, object]] = {}
    results_rows: List[Dict[str, object]] = []
    leaderboard_totals: Dict[str, Dict[str, float]] = {}

    for denorm in denorms:
        models = build_database_models(schemas, stats, denorm)
        embeds = collect_embed_paths(denorm)

        print(f"Denormalization {denorm.id}: {denorm.description}")
        for query in queries:
            plan = plan_query(query, models, embeds)
            result = simulate_plan(plan, models, cluster)

            per_query_plan_json.setdefault(query.id, {})[denorm.id] = plan_to_json(
                query, {denorm.id: result}
            )["per_denorm"][denorm.id]

            weighted_time = result.total_cost.time_cost * query.frequency
            weighted_carbon = result.total_cost.carbon_cost * query.frequency
            weighted_price = result.total_cost.price_cost * query.frequency
            totals = leaderboard_totals.setdefault(
                denorm.id, {"time": 0.0, "carbon": 0.0, "price": 0.0}
            )
            totals["time"] += weighted_time
            totals["carbon"] += weighted_carbon
            totals["price"] += weighted_price

            results_rows.append(
                {
                    "denorm_id": denorm.id,
                    "query_id": query.id,
                    "operator_plan_summary": operator_plan_summary(result),
                    "time": result.total_cost.time_cost,
                    "carbon": result.total_cost.carbon_cost,
                    "price": result.total_cost.price_cost,
                    "scanned_docs": result.scanned_docs,
                    "output_docs": result.output_docs,
                    "scanned_bytes": result.scanned_bytes,
                    "returned_bytes": result.output_bytes,
                }
            )

            print(summarize_result(result))
        print("")

    for query_id, per_denorm in per_query_plan_json.items():
        file_stub = query_id.lower()
        if query_id.upper().startswith("Q") and query_id[1:].isdigit():
            file_stub = f"query{query_id[1:]}"
        plan_path = plans_dir / f"{file_stub}.plan.json"
        plan_path.write_text(json.dumps({"query_id": query_id, "per_denorm": per_denorm}, indent=2))

    ordering = sorted(
        leaderboard_totals,
        key=lambda name: (
            leaderboard_totals[name]["price"],
            leaderboard_totals[name]["carbon"],
            leaderboard_totals[name]["time"],
        ),
    )
    leaderboard_path = out_dir / "leaderboard.md"
    leaderboard_path.write_text(leaderboard_md(leaderboard_totals, ordering), encoding="utf-8")

    results_path = out_dir / "results.csv"
    write_results_csv(results_path, results_rows)

    print("Leaderboard written to:", leaderboard_path)
    print("Results written to:", results_path)


if __name__ == "__main__":
    run()
