"""
QuerySimulator package.

Provides cost estimation utilities for filter and join queries (Q1-Q5) using
a simple IO-based cost model (time, carbon, and price).
"""

from .config import (
    STATS,
    DOC_SIZES_BYTES,
    COLLECTION_SIZES_GB,
    collection_size_gb,
    BYTES_PER_GB,
    BASE_IO_TIME_UNIT,
    BASE_IO_CARBON_UNIT,
    BASE_IO_PRICE_UNIT,
    NETWORK_MULTIPLIER,
    SHARDING_ACCESS_FRACTION,
)
from .models import QuerySpec, QueryComponent, CostBreakdown, QueryCostResult, AggregateResult
from .costs import QueryCostModel
from .queries import default_queries
from .runner import simulate_queries, format_cost_results
from .aggregate import (
    aggregate_with_sharding,
    aggregate_without_sharding,
    estimate_filter_selectivity,
    estimate_group_cardinality,
    estimate_shuffle_volume_gb,
    estimate_reduce_volume_gb,
)

__all__ = [
    "STATS",
    "DOC_SIZES_BYTES",
    "COLLECTION_SIZES_GB",
    "collection_size_gb",
    "BYTES_PER_GB",
    "BASE_IO_TIME_UNIT",
    "BASE_IO_CARBON_UNIT",
    "BASE_IO_PRICE_UNIT",
    "NETWORK_MULTIPLIER",
    "SHARDING_ACCESS_FRACTION",
    "QuerySpec",
    "QueryComponent",
    "CostBreakdown",
    "QueryCostResult",
    "AggregateResult",
    "QueryCostModel",
    "default_queries",
    "simulate_queries",
    "format_cost_results",
    "aggregate_with_sharding",
    "aggregate_without_sharding",
    "estimate_filter_selectivity",
    "estimate_group_cardinality",
    "estimate_shuffle_volume_gb",
    "estimate_reduce_volume_gb",
]
