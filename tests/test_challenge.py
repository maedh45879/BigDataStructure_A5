from pathlib import Path

import pytest

from challenge.config import BASE_IO_TIME_UNIT, BYTES_PER_GB, KEY_OVERHEAD_BYTES
from challenge.loader import (
    build_database_models,
    collect_embed_paths,
    load_denormalizations,
    load_schema,
    load_stats,
)
from challenge.models import (
    ClusterConfig,
    CollectionConfig,
    CollectionModel,
    CollectionSchema,
    CollectionStats,
    FieldSpec,
    FilterPredicate,
    PlanOperator,
    QueryPlan,
    QuerySpec,
)
from challenge.planner import plan_query
from challenge.simulator import simulate_plan


BASE_DIR = Path(__file__).resolve().parents[1]


def test_translate_query_plans() -> None:
    schemas = load_schema(BASE_DIR / "challenge" / "schema.json")
    _, stats, _ = load_stats(BASE_DIR / "challenge" / "stats.json")
    denorms = load_denormalizations(BASE_DIR / "challenge" / "denormalizations.json")
    queries = [
        QuerySpec(
            id=item["id"],
            sql=item["sql"],
            frequency=item.get("frequency", 1.0),
        )
        for item in [
            {"id": "Q1", "sql": "SELECT description FROM Product WHERE categorie = 'smartphone';"},
            {
                "id": "Q2",
                "sql": "SELECT ol.quantity, p.price FROM OrderLine ol JOIN Product p ON ol.IDP = p.IDP WHERE p.brand = 'apple' AND ol.IDC = 125;",
            },
        ]
    ]

    denorm = next(item for item in denorms if item.id == "D1")
    models = build_database_models(schemas, stats, denorm)
    embeds = collect_embed_paths(denorm)

    q1_plan = plan_query(queries[0], models, embeds)
    assert len(q1_plan.operators) == 1
    assert q1_plan.operators[0].target_collection == "Product"

    q2_plan = plan_query(queries[1], models, embeds)
    assert any(op.operator_type.startswith("nested_loop") for op in q2_plan.operators)


def test_cost_engine_deterministic_filter() -> None:
    schema = CollectionSchema(
        name="Foo",
        primary_key="id",
        fields={
            "id": FieldSpec(name="id", avg_size=8),
            "value": FieldSpec(name="value", avg_size=10),
        },
    )
    stats = CollectionStats(
        nb_documents=100,
        distinct_values={"value": 10},
    )
    config = CollectionConfig(sharding_key="id", indexes=[])
    collection = CollectionModel(schema=schema, stats=stats, config=config)

    plan = QueryPlan(
        query=QuerySpec(id="QX", sql="SELECT value FROM Foo WHERE value = 1;"),
        involved_collections=["Foo"],
        operators=[
            PlanOperator(
                name="QX_filter",
                operator_type="filter_without_sharding",
                target_collection="Foo",
                filters=[FilterPredicate(collection="Foo", field="value", value=1)],
                output_fields=["value"],
                scan_strategy="full",
            )
        ],
    )

    result = simulate_plan(plan, {"Foo": collection}, cluster=ClusterConfig(nb_servers=10, sharding_access_fraction=0.1))
    expected_scanned_bytes = 100 * (8 + 10)
    expected_data_gb = expected_scanned_bytes / BYTES_PER_GB
    assert result.scanned_bytes == expected_scanned_bytes
    assert result.total_cost.time_cost == pytest.approx(expected_data_gb * BASE_IO_TIME_UNIT)
    assert result.output_bytes == 10 * (KEY_OVERHEAD_BYTES + 10)


def test_denormalization_removes_join() -> None:
    schemas = load_schema(BASE_DIR / "challenge" / "schema.json")
    _, stats, _ = load_stats(BASE_DIR / "challenge" / "stats.json")
    denorms = load_denormalizations(BASE_DIR / "challenge" / "denormalizations.json")
    denorm = next(item for item in denorms if item.id == "D2")
    models = build_database_models(schemas, stats, denorm)
    embeds = collect_embed_paths(denorm)
    query = QuerySpec(
        id="Q2",
        sql="SELECT ol.quantity, p.price FROM OrderLine ol JOIN Product p ON ol.IDP = p.IDP WHERE p.brand = 'apple' AND ol.IDC = 125;",
    )

    plan = plan_query(query, models, embeds)
    assert len(plan.operators) == 1
    assert plan.operators[0].operator_type.startswith("filter")
    assert plan.operators[0].target_collection == "OrderLine"
