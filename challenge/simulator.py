from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .config import BYTES_PER_GB, KEY_OVERHEAD_BYTES
from .costs import CostModel
from .models import (
    ClusterConfig,
    CollectionModel,
    FilterPredicate,
    JoinPredicate,
    OperatorMetrics,
    PlanOperator,
    QueryPlan,
    QueryResult,
)


def _selectivity(stats: CollectionModel, field: str) -> float:
    if field in stats.stats.field_selectivity:
        return stats.stats.field_selectivity[field]
    distinct = stats.stats.distinct_values.get(field)
    if distinct and distinct > 0:
        return 1.0 / distinct
    return 1.0


def _output_doc_size(
    collection: CollectionModel,
    output_fields: List[str],
) -> int:
    if not output_fields:
        return collection.document_size_bytes()
    total = 0
    for field in output_fields:
        total += KEY_OVERHEAD_BYTES + collection.schema.field_size_bytes(
            field, collection.stats.avg_array_lengths
        )
    return total


def _filter_metrics(
    operator: PlanOperator,
    collection: CollectionModel,
    cluster: ClusterConfig,
    model: CostModel,
) -> OperatorMetrics:
    selectivity = 1.0
    for predicate in operator.filters:
        selectivity *= _selectivity(collection, predicate.field)

    base_docs = collection.stats.nb_documents
    output_docs = max(0, int(base_docs * selectivity))
    if base_docs > 0 and selectivity > 0 and output_docs == 0:
        output_docs = 1

    sharding_fraction = (
        cluster.sharding_access_fraction
        if operator.scan_strategy == "shard"
        else 1.0
    )
    if operator.scan_strategy == "index":
        scanned_docs = output_docs
    else:
        scanned_docs = max(0, int(base_docs * sharding_fraction))
        if scanned_docs == 0 and base_docs > 0:
            scanned_docs = 1

    doc_size = collection.document_size_bytes()
    scanned_bytes = scanned_docs * doc_size

    output_doc_size = _output_doc_size(collection, operator.output_fields)
    output_bytes = output_docs * output_doc_size

    data_gb = scanned_bytes / BYTES_PER_GB
    cost = model.io_cost(f"{operator.name}:filter", data_gb, use_network=False)

    return OperatorMetrics(
        operator=operator,
        scanned_docs=scanned_docs,
        output_docs=output_docs,
        scanned_bytes=scanned_bytes,
        output_bytes=output_bytes,
        shuffled_bytes=0,
        output_doc_size_bytes=output_doc_size,
        cost=cost,
        details=[cost],
    )


def _resolve_output_field(
    field: str,
    left: CollectionModel,
    right: CollectionModel,
) -> Tuple[CollectionModel, str]:
    if "." in field:
        collection, name = field.split(".", 1)
        if collection == left.schema.name:
            return left, name
        if collection == right.schema.name:
            return right, name
    return left, field


def _join_output_size(
    fields: List[str],
    left: CollectionModel,
    right: CollectionModel,
) -> int:
    if not fields:
        return left.document_size_bytes() + right.document_size_bytes()
    total = 0
    for field in fields:
        target, name = _resolve_output_field(field, left, right)
        total += KEY_OVERHEAD_BYTES + target.schema.field_size_bytes(
            name, target.stats.avg_array_lengths
        )
    return total


def _estimate_join_selectivity(left: CollectionModel, right: CollectionModel, join: JoinPredicate) -> float:
    left_card = left.stats.distinct_values.get(join.left_field)
    right_card = right.stats.distinct_values.get(join.right_field)
    cardinality = max(left_card or 0, right_card or 0, 1)
    return 1.0 / cardinality


def _estimate_group_cardinality(
    collection: CollectionModel,
    grouping_keys: List[str],
    input_docs: int,
) -> int:
    total = 1
    for key in grouping_keys:
        cardinality = collection.stats.distinct_values.get(key)
        if not cardinality:
            cardinality = input_docs
        total *= cardinality
        if total >= input_docs:
            break
    return max(1, min(int(total), input_docs))


def _aggregate_metrics(
    operator: PlanOperator,
    collection: CollectionModel,
    cluster: ClusterConfig,
    model: CostModel,
) -> OperatorMetrics:
    if not operator.grouping_keys:
        raise ValueError("Aggregate operator requires grouping keys.")

    filter_selectivity = 1.0
    if operator.filters:
        for predicate in operator.filters:
            filter_selectivity *= _selectivity(collection, predicate.field)

    base_docs = collection.stats.nb_documents
    input_docs = max(1, int(base_docs * filter_selectivity)) if base_docs > 0 else 0
    output_docs = _estimate_group_cardinality(collection, operator.grouping_keys, input_docs)

    output_doc_size = _output_doc_size(collection, operator.output_fields or operator.grouping_keys)
    output_bytes = output_docs * output_doc_size

    sharding_fraction = (
        cluster.sharding_access_fraction
        if operator.use_sharding
        and collection.config.sharding_key in operator.grouping_keys
        else 1.0
    )
    scan_bytes = int(collection.document_size_bytes() * input_docs * sharding_fraction)

    shuffle_bytes = 0
    if not (operator.use_sharding and collection.config.sharding_key in operator.grouping_keys):
        shuffle_bytes = output_bytes * cluster.nb_servers

    map_cost = model.io_cost(f"{operator.name}:map", scan_bytes / BYTES_PER_GB)
    shuffle_cost = model.io_cost(
        f"{operator.name}:shuffle",
        shuffle_bytes / BYTES_PER_GB,
        use_network=shuffle_bytes > 0,
    )
    reduce_cost = model.io_cost(
        f"{operator.name}:reduce",
        (shuffle_bytes + output_bytes) / BYTES_PER_GB,
    )
    total = model.aggregate(f"{operator.name}:aggregate_total", [map_cost, shuffle_cost, reduce_cost])

    return OperatorMetrics(
        operator=operator,
        scanned_docs=input_docs,
        output_docs=output_docs,
        scanned_bytes=scan_bytes + shuffle_bytes,
        output_bytes=output_bytes,
        shuffled_bytes=shuffle_bytes,
        output_doc_size_bytes=output_doc_size,
        cost=total,
        details=[map_cost, shuffle_cost, reduce_cost],
    )


def _join_metrics(
    operator: PlanOperator,
    left_metrics: Optional[OperatorMetrics],
    right_metrics: Optional[OperatorMetrics],
    left: CollectionModel,
    right: CollectionModel,
    model: CostModel,
) -> OperatorMetrics:
    left_docs = left_metrics.output_docs if left_metrics else left.stats.nb_documents
    right_docs = right_metrics.output_docs if right_metrics else right.stats.nb_documents

    join_selectivity = _estimate_join_selectivity(left, right, operator.join)
    output_docs = max(0, int(min(left_docs, right_docs) * join_selectivity))
    if output_docs == 0 and left_docs > 0 and right_docs > 0:
        output_docs = 1

    left_bytes = left_metrics.output_bytes if left_metrics else left_docs * left.document_size_bytes()
    right_bytes = right_metrics.output_bytes if right_metrics else right_docs * right.document_size_bytes()
    scan_bytes = left_bytes + right_bytes

    shuffle_bytes = 0
    if operator.operator_type == "nested_loop_without_sharding":
        shuffle_bytes = scan_bytes

    output_doc_size = _join_output_size(operator.output_fields, left, right)
    output_bytes = output_docs * output_doc_size

    scan_cost = model.io_cost(f"{operator.name}:join_scan", scan_bytes / BYTES_PER_GB)
    shuffle_cost = model.io_cost(
        f"{operator.name}:join_shuffle",
        shuffle_bytes / BYTES_PER_GB,
        use_network=shuffle_bytes > 0,
    )
    total = model.aggregate(f"{operator.name}:join_total", [scan_cost, shuffle_cost])

    return OperatorMetrics(
        operator=operator,
        scanned_docs=left_docs + right_docs,
        output_docs=output_docs,
        scanned_bytes=scan_bytes + shuffle_bytes,
        output_bytes=output_bytes,
        shuffled_bytes=shuffle_bytes,
        output_doc_size_bytes=output_doc_size,
        cost=total,
        details=[scan_cost, shuffle_cost],
    )


def simulate_plan(
    plan: QueryPlan,
    collections: Dict[str, CollectionModel],
    cluster: ClusterConfig,
    model: Optional[CostModel] = None,
) -> QueryResult:
    model = model or CostModel()
    results: List[OperatorMetrics] = []
    outputs: Dict[str, OperatorMetrics] = {}

    for operator in plan.operators:
        if operator.operator_type.startswith("filter"):
            collection = collections[operator.target_collection]
            metrics = _filter_metrics(operator, collection, cluster, model)
            outputs[operator.name] = metrics
            results.append(metrics)
            continue

        if operator.operator_type.startswith("nested_loop"):
            left = collections[operator.left_collection]
            right = collections[operator.right_collection]
            left_metrics = outputs.get(f"{plan.query.id}_filter_{left.schema.name}")
            right_metrics = outputs.get(f"{plan.query.id}_filter_{right.schema.name}")
            metrics = _join_metrics(operator, left_metrics, right_metrics, left, right, model)
            outputs[operator.name] = metrics
            results.append(metrics)
            continue

        if operator.operator_type.startswith("aggregate"):
            collection = collections[operator.target_collection]
            metrics = _aggregate_metrics(operator, collection, cluster, model)
            outputs[operator.name] = metrics
            results.append(metrics)
            continue

        raise ValueError(f"Unsupported operator type: {operator.operator_type}")

    if results:
        total = model.aggregate(
            f"{plan.query.id}:total", [metric.cost for metric in results]
        )
        scanned_docs = sum(metric.scanned_docs for metric in results)
        scanned_bytes = sum(metric.scanned_bytes for metric in results)
        shuffled_bytes = sum(metric.shuffled_bytes for metric in results)
        output_docs = results[-1].output_docs
        output_bytes = results[-1].output_bytes
    else:
        total = model.io_cost(f"{plan.query.id}:total", 0.0)
        scanned_docs = 0
        output_docs = 0
        scanned_bytes = 0
        output_bytes = 0
        shuffled_bytes = 0

    return QueryResult(
        plan=plan,
        operators=results,
        total_cost=total,
        scanned_docs=scanned_docs,
        output_docs=output_docs,
        scanned_bytes=scanned_bytes,
        output_bytes=output_bytes,
        shuffled_bytes=shuffled_bytes,
    )
