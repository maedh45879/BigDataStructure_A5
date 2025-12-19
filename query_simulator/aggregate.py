from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .config import BYTES_PER_GB, SHARDING_ACCESS_FRACTION, STATS
from .costs import QueryCostModel
from .models import AggregateResult, CostBreakdown
from main import CollectionModel, CollectionStats, compute_document_size_bytes


DEFAULT_UNKNOWN_FIELD_SIZE_BYTES = 8
KEY_OVERHEAD_BYTES = 12


@dataclass
class AggregateSizing:
    output_documents: int
    output_doc_size_bytes: int
    output_size_gb: float
    missing_fields: List[str]


def estimate_filter_selectivity(stats: CollectionStats, filtered_key: Optional[str]) -> float:
    """Estimate selectivity for an equality filter on a single key."""
    if filtered_key is None:
        return 1.0
    if filtered_key in stats.field_selectivity:
        return stats.field_selectivity[filtered_key]
    cardinality = estimate_key_cardinality(stats, filtered_key)
    if cardinality <= 0:
        raise ValueError(f"Invalid cardinality for filtered key: {filtered_key}")
    return 1.0 / cardinality


def estimate_key_cardinality(stats: CollectionStats, key: str) -> int:
    """Retrieve cardinality for a grouping/filter key from stats."""
    if key in stats.field_cardinality:
        return stats.field_cardinality[key]
    if key in stats.sharding_key_cardinality:
        return stats.sharding_key_cardinality[key]
    raise ValueError(f"Missing cardinality for key: {key}")


def estimate_group_cardinality(
    stats: CollectionStats,
    grouping_keys: Sequence[str],
    input_documents: int,
) -> int:
    """Estimate distinct group count, capped by the number of input documents."""
    if not grouping_keys:
        raise ValueError("Grouping keys are required for aggregation.")
    total = 1
    for key in grouping_keys:
        total *= estimate_key_cardinality(stats, key)
        if total >= input_documents:
            break
    total = min(total, input_documents)
    return max(total, 0)


def resolve_field_schema(schema: dict, field_path: str) -> Optional[dict]:
    """Resolve a field path (dot-separated) to a JSON schema fragment."""
    current: Optional[dict] = schema
    for part in field_path.split("."):
        if current is None:
            return None
        if current.get("type") == "array":
            current = current.get("items", {})
        if current.get("type") != "object":
            return None
        current = current.get("properties", {}).get(part)
    return current


def estimate_field_size_bytes(
    schema: dict,
    stats: CollectionStats,
    field_path: str,
) -> Tuple[int, bool]:
    """Estimate size of a field from schema; returns size and missing flag."""
    field_schema = resolve_field_schema(schema, field_path)
    if field_schema is None:
        return DEFAULT_UNKNOWN_FIELD_SIZE_BYTES, True
    prefix = f"{field_path}."
    return compute_document_size_bytes(field_schema, stats, prefix=prefix), False


def estimate_output_doc_size_bytes(
    schema: dict,
    stats: CollectionStats,
    output_fields: Iterable[str],
) -> AggregateSizing:
    """Estimate output document size for a list of output fields."""
    total = 0
    missing_fields: List[str] = []
    for field in output_fields:
        field_size, is_missing = estimate_field_size_bytes(schema, stats, field)
        total += KEY_OVERHEAD_BYTES + field_size
        if is_missing:
            missing_fields.append(field)
    return AggregateSizing(
        output_documents=0,
        output_doc_size_bytes=total,
        output_size_gb=0.0,
        missing_fields=missing_fields,
    )


def estimate_shuffle_volume_gb(
    group_cardinality: int,
    partial_doc_size_bytes: int,
    shards_touched: int,
    aligned_with_sharding: bool,
) -> float:
    """Estimate shuffle data volume in GB for a Map/Reduce aggregation."""
    if aligned_with_sharding:
        return 0.0
    total_bytes = group_cardinality * shards_touched * partial_doc_size_bytes
    return total_bytes / BYTES_PER_GB


def estimate_reduce_volume_gb(shuffle_gb: float, output_gb: float) -> float:
    """Estimate reduce IO volume as shuffle read + output write."""
    return shuffle_gb + output_gb


def aggregate_with_sharding(
    collection: CollectionModel,
    grouping_keys: Sequence[str],
    output_fields: Sequence[str],
    filtered_key: Optional[str] = None,
    model: Optional[QueryCostModel] = None,
    label: str = "aggregate_with_sharding",
) -> AggregateResult:
    return _aggregate(
        collection=collection,
        grouping_keys=grouping_keys,
        output_fields=output_fields,
        filtered_key=filtered_key,
        model=model,
        label=label,
        use_sharding=True,
    )


def aggregate_without_sharding(
    collection: CollectionModel,
    grouping_keys: Sequence[str],
    output_fields: Sequence[str],
    filtered_key: Optional[str] = None,
    model: Optional[QueryCostModel] = None,
    label: str = "aggregate_without_sharding",
) -> AggregateResult:
    return _aggregate(
        collection=collection,
        grouping_keys=grouping_keys,
        output_fields=output_fields,
        filtered_key=filtered_key,
        model=model,
        label=label,
        use_sharding=False,
    )


def _aggregate(
    collection: CollectionModel,
    grouping_keys: Sequence[str],
    output_fields: Sequence[str],
    filtered_key: Optional[str],
    model: Optional[QueryCostModel],
    label: str,
    use_sharding: bool,
) -> AggregateResult:
    model = model or QueryCostModel()
    stats = collection.stats
    notes: List[str] = []

    filter_selectivity = estimate_filter_selectivity(stats, filtered_key)
    if filtered_key is not None:
        notes.append(f"Filter on {filtered_key} (selectivity {filter_selectivity:.6f})")

    base_docs = stats.nb_documents
    if base_docs <= 0 or filter_selectivity <= 0:
        input_docs = 0
    else:
        input_docs = max(1, int(base_docs * filter_selectivity))

    group_cardinality = estimate_group_cardinality(stats, grouping_keys, input_docs)
    output_documents = min(group_cardinality, input_docs)

    output_fields_list = list(output_fields) or list(grouping_keys)
    sizing = estimate_output_doc_size_bytes(collection.schema, stats, output_fields_list)
    output_doc_size_bytes = sizing.output_doc_size_bytes
    output_size_gb = output_documents * output_doc_size_bytes / BYTES_PER_GB

    if sizing.missing_fields:
        notes.append(
            "Unknown output fields sized as 8B primitives: "
            + ", ".join(sizing.missing_fields)
        )

    targeted_shard = (
        use_sharding
        and stats.sharding_key is not None
        and filtered_key == stats.sharding_key
    )
    if targeted_shard:
        notes.append(
            f"Targeted shard access (x{SHARDING_ACCESS_FRACTION:.3f} of cluster)"
        )

    scan_fraction = filter_selectivity
    if targeted_shard:
        scan_fraction *= SHARDING_ACCESS_FRACTION
    map_scan_gb = collection.size_gb() * scan_fraction

    aligned_with_sharding = (
        use_sharding
        and stats.sharding_key is not None
        and set(grouping_keys) == {stats.sharding_key}
    )
    if aligned_with_sharding:
        notes.append("Grouping aligns with sharding key; shuffle avoided.")

    shard_count = int(STATS.get("N_SERVERS", 1))
    if targeted_shard:
        shards_touched = max(1, int(shard_count * SHARDING_ACCESS_FRACTION))
    else:
        shards_touched = shard_count

    shuffle_gb = estimate_shuffle_volume_gb(
        group_cardinality=output_documents,
        partial_doc_size_bytes=output_doc_size_bytes,
        shards_touched=shards_touched,
        aligned_with_sharding=aligned_with_sharding,
    )

    reduce_gb = estimate_reduce_volume_gb(shuffle_gb=shuffle_gb, output_gb=output_size_gb)

    map_cost = model.io_cost(
        label=f"{label}:map",
        data_gb=map_scan_gb,
        use_network=False,
        notes=["Map scan + local grouping"],
    )
    shuffle_cost = model.io_cost(
        label=f"{label}:shuffle",
        data_gb=shuffle_gb,
        use_network=shuffle_gb > 0,
        notes=["Shuffle partial aggregates"],
    )
    reduce_cost = model.io_cost(
        label=f"{label}:reduce",
        data_gb=reduce_gb,
        use_network=False,
        notes=["Reduce + final output write"],
    )
    total_cost = model.aggregate(f"{label}:total", [map_cost, shuffle_cost, reduce_cost])

    return AggregateResult(
        label=label,
        collection=collection.name,
        grouping_keys=list(grouping_keys),
        output_fields=output_fields_list,
        filtered_key=filtered_key,
        output_documents=output_documents,
        output_size_gb=output_size_gb,
        map_cost=map_cost,
        shuffle_cost=shuffle_cost,
        reduce_cost=reduce_cost,
        total_cost=total_cost,
        notes=notes,
    )
