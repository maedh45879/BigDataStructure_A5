from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from main import CollectionModel, DatabaseModel
from query_simulator.aggregate import (
    DEFAULT_UNKNOWN_FIELD_SIZE_BYTES,
    KEY_OVERHEAD_BYTES,
    aggregate_with_sharding,
    aggregate_without_sharding,
    estimate_field_size_bytes,
    estimate_filter_selectivity,
)
from query_simulator.config import BYTES_PER_GB, SHARDING_ACCESS_FRACTION
from query_simulator.costs import QueryCostModel
from query_simulator.models import CostBreakdown

from .loader import ClusterConfig
from .models import PlanStep, PlanResult, QueryPlan, StepResult


def _resolve_input(
    ref: str,
    db_model: DatabaseModel,
    step_results: Dict[str, StepResult],
) -> Tuple[str, CollectionModel | StepResult]:
    if ref in step_results:
        return "step", step_results[ref]
    if ref in db_model.collections:
        return "collection", db_model.collections[ref]
    raise ValueError(f"Unknown input reference: {ref}")


def _estimate_field_sizes_from_schema(
    collection: CollectionModel, fields: List[str]
) -> Tuple[Dict[str, int], List[str]]:
    field_sizes: Dict[str, int] = {}
    missing_fields: List[str] = []
    for field in fields:
        size, missing = estimate_field_size_bytes(collection.schema, collection.stats, field)
        field_sizes[field] = size
        if missing:
            missing_fields.append(field)
    return field_sizes, missing_fields


def _estimate_field_sizes_from_step(
    step: StepResult, fields: List[str]
) -> Tuple[Dict[str, int], List[str]]:
    field_sizes: Dict[str, int] = {}
    missing_fields: List[str] = []
    for field in fields:
        size = step.field_sizes.get(field)
        if size is None:
            size = DEFAULT_UNKNOWN_FIELD_SIZE_BYTES
            missing_fields.append(field)
        field_sizes[field] = size
    return field_sizes, missing_fields


def _resolve_output_fields(
    fields: List[str],
    left: CollectionModel | StepResult,
    right: Optional[CollectionModel | StepResult] = None,
) -> Tuple[Dict[str, int], List[str]]:
    field_sizes: Dict[str, int] = {}
    missing_fields: List[str] = []
    for field in fields:
        source = left
        field_name = field
        if field.startswith("right.") and right is not None:
            source = right
            field_name = field.split("right.", 1)[1]
        elif field.startswith("left."):
            source = left
            field_name = field.split("left.", 1)[1]

        if isinstance(source, CollectionModel):
            size, missing = estimate_field_size_bytes(source.schema, source.stats, field_name)
            field_sizes[field] = size
            if missing:
                missing_fields.append(field)
        else:
            size = source.field_sizes.get(field_name)
            if size is None:
                size = DEFAULT_UNKNOWN_FIELD_SIZE_BYTES
                missing_fields.append(field)
            field_sizes[field] = size
    return field_sizes, missing_fields


def _output_doc_size_bytes(field_sizes: Dict[str, int]) -> int:
    return sum(KEY_OVERHEAD_BYTES + size for size in field_sizes.values())


def _scale_docs(count: int, selectivity: float) -> int:
    if count <= 0 or selectivity <= 0:
        return 0
    return max(1, int(count * selectivity))


def _simulate_filter(
    step: PlanStep,
    input_obj: CollectionModel | StepResult,
    model: QueryCostModel,
) -> StepResult:
    notes: List[str] = []
    if isinstance(input_obj, CollectionModel):
        base_docs = input_obj.stats.nb_documents
        base_size_gb = input_obj.size_gb()
        stats = input_obj.stats
    else:
        base_docs = input_obj.output_documents
        base_size_gb = input_obj.output_size_gb
        stats = None

    selectivity = step.selectivity
    if selectivity is None:
        if stats is None:
            raise ValueError(
                f"Filter step '{step.name}' needs explicit selectivity for intermediate input."
            )
        selectivity = estimate_filter_selectivity(stats, step.filter_key)

    targeted_shard = False
    if step.use_sharding and stats is not None and step.filter_key == stats.sharding_key:
        targeted_shard = True
        notes.append(f"Targeted shard access x{SHARDING_ACCESS_FRACTION:.3f}")

    scan_fraction = selectivity * (SHARDING_ACCESS_FRACTION if targeted_shard else 1.0)
    data_gb = base_size_gb * scan_fraction

    cost = model.io_cost(
        label=f"{step.name}:filter",
        data_gb=data_gb,
        use_network=False,
        notes=notes,
    )

    output_docs = _scale_docs(base_docs, selectivity)

    if step.output_fields:
        if isinstance(input_obj, CollectionModel):
            field_sizes, missing = _estimate_field_sizes_from_schema(
                input_obj, step.output_fields
            )
        else:
            field_sizes, missing = _estimate_field_sizes_from_step(
                input_obj, step.output_fields
            )
        if missing:
            notes.append(
                "Unknown fields sized as 8B: " + ", ".join(sorted(set(missing)))
            )
    else:
        field_sizes = {}
    doc_size_bytes = (
        _output_doc_size_bytes(field_sizes)
        if field_sizes
        else int((base_size_gb * BYTES_PER_GB) / max(base_docs, 1))
    )
    output_size_gb = output_docs * doc_size_bytes / BYTES_PER_GB

    return StepResult(
        step=step,
        output_documents=output_docs,
        output_doc_size_bytes=doc_size_bytes,
        output_size_gb=output_size_gb,
        cost=cost,
        details=[cost],
        field_sizes=field_sizes,
        notes=notes,
        sharding_key=step.filter_key if targeted_shard else None,
    )


def _simulate_join(
    step: PlanStep,
    left: CollectionModel | StepResult,
    right: CollectionModel | StepResult,
    model: QueryCostModel,
) -> StepResult:
    notes: List[str] = []
    if isinstance(left, CollectionModel):
        left_docs = left.stats.nb_documents
        left_size_gb = left.size_gb()
        left_shard = left.stats.sharding_key
    else:
        left_docs = left.output_documents
        left_size_gb = left.output_size_gb
        left_shard = left.sharding_key

    if isinstance(right, CollectionModel):
        right_docs = right.stats.nb_documents
        right_size_gb = right.size_gb()
        right_shard = right.stats.sharding_key
    else:
        right_docs = right.output_documents
        right_size_gb = right.output_size_gb
        right_shard = right.sharding_key

    join_selectivity = step.join_selectivity or 1.0
    output_docs = _scale_docs(min(left_docs, right_docs), join_selectivity)

    base_scan_gb = left_size_gb + right_size_gb
    scan_left = model.io_cost(f"{step.name}:scan_left", left_size_gb)
    scan_right = model.io_cost(f"{step.name}:scan_right", right_size_gb)

    aligned = (
        step.use_sharding
        and step.join_key is not None
        and step.join_key == left_shard
        and step.join_key == right_shard
    )
    shuffle_gb = 0.0 if aligned else base_scan_gb
    if aligned:
        notes.append("Join key aligned with sharding; shuffle avoided.")

    shuffle = model.io_cost(
        f"{step.name}:shuffle", shuffle_gb, use_network=shuffle_gb > 0
    )
    total = model.aggregate(f"{step.name}:total", [scan_left, scan_right, shuffle])

    field_sizes, missing = _resolve_output_fields(step.output_fields, left, right)
    if missing:
        notes.append("Unknown fields sized as 8B: " + ", ".join(sorted(set(missing))))
    doc_size_bytes = _output_doc_size_bytes(field_sizes)
    output_size_gb = output_docs * doc_size_bytes / BYTES_PER_GB

    return StepResult(
        step=step,
        output_documents=output_docs,
        output_doc_size_bytes=doc_size_bytes,
        output_size_gb=output_size_gb,
        cost=total,
        details=[scan_left, scan_right, shuffle],
        field_sizes=field_sizes,
        notes=notes,
        sharding_key=step.join_key if aligned else None,
    )


def _simulate_aggregate(
    step: PlanStep,
    collection: CollectionModel,
    model: QueryCostModel,
) -> StepResult:
    if step.use_sharding:
        result = aggregate_with_sharding(
            collection,
            grouping_keys=step.grouping_keys,
            output_fields=step.output_fields,
            filtered_key=step.filter_key,
            model=model,
            label=step.name,
        )
    else:
        result = aggregate_without_sharding(
            collection,
            grouping_keys=step.grouping_keys,
            output_fields=step.output_fields,
            filtered_key=step.filter_key,
            model=model,
            label=step.name,
        )

    field_sizes, missing = _estimate_field_sizes_from_schema(collection, result.output_fields)
    notes = list(result.notes)
    if missing:
        notes.append("Unknown fields sized as 8B: " + ", ".join(sorted(set(missing))))

    if result.output_documents > 0:
        doc_size_bytes = int(result.output_size_gb * BYTES_PER_GB / result.output_documents)
    else:
        doc_size_bytes = 0

    return StepResult(
        step=step,
        output_documents=result.output_documents,
        output_doc_size_bytes=doc_size_bytes,
        output_size_gb=result.output_size_gb,
        cost=result.total_cost,
        details=[result.map_cost, result.shuffle_cost, result.reduce_cost],
        field_sizes=field_sizes,
        notes=notes,
        sharding_key=step.grouping_keys[0] if len(step.grouping_keys) == 1 else None,
    )


def simulate_plan(
    plan: QueryPlan,
    db_model: DatabaseModel,
    cluster_config: Optional[ClusterConfig] = None,
    model: Optional[QueryCostModel] = None,
) -> PlanResult:
    model = model or QueryCostModel()
    cluster_config = cluster_config or ClusterConfig()
    step_results: Dict[str, StepResult] = {}
    results: List[StepResult] = []

    for step in plan.steps:
        if step.operator_type == "filter":
            ref = step.input_ref or step.target_collection
            if not ref:
                raise ValueError(f"Filter step '{step.name}' missing input reference.")
            _, input_obj = _resolve_input(ref, db_model, step_results)
            result = _simulate_filter(step, input_obj, model)
        elif step.operator_type == "join":
            if not step.left_ref or not step.right_ref:
                raise ValueError(f"Join step '{step.name}' missing inputs.")
            _, left = _resolve_input(step.left_ref, db_model, step_results)
            _, right = _resolve_input(step.right_ref, db_model, step_results)
            result = _simulate_join(step, left, right, model)
        elif step.operator_type == "aggregate":
            ref = step.input_ref or step.target_collection
            if not ref:
                raise ValueError(f"Aggregate step '{step.name}' missing input reference.")
            kind, input_obj = _resolve_input(ref, db_model, step_results)
            if kind != "collection":
                raise ValueError(
                    f"Aggregate step '{step.name}' only supports base collections."
                )
            result = _simulate_aggregate(step, input_obj, model)
        else:
            raise ValueError(f"Unsupported operator: {step.operator_type}")

        step_results[step.name] = result
        results.append(result)

    if results:
        total = model.aggregate(
            f"{plan.query.name}:total", [step.cost for step in results]
        )
        output_docs = results[-1].output_documents
        output_size_gb = results[-1].output_size_gb
    else:
        total = model.io_cost(f"{plan.query.name}:total", data_gb=0.0)
        output_docs = 0
        output_size_gb = 0.0

    return PlanResult(
        plan=plan,
        steps=results,
        total_cost=total,
        output_documents=output_docs,
        output_size_gb=output_size_gb,
    )
