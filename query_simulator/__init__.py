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
from .models import QuerySpec, QueryComponent, CostBreakdown, QueryCostResult
from .costs import QueryCostModel
from .queries import default_queries
from .runner import simulate_queries, format_cost_results

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
    "QueryCostModel",
    "default_queries",
    "simulate_queries",
    "format_cost_results",
]
