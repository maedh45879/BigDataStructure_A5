from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


QueryKind = Literal["filter", "join"]


@dataclass
class QueryComponent:
    """Represents one collection touched by a query."""

    name: str
    collection: str
    selectivity: float
    description: str
    targeted_shard: bool = False
    requires_network: bool = False


@dataclass
class QuerySpec:
    """High-level description of a query (filter or join)."""

    name: str
    kind: QueryKind
    description: str
    components: List[QueryComponent]
    join_selectivity: Optional[float] = None


@dataclass
class CostBreakdown:
    label: str
    data_scanned_gb: float
    time_cost: float
    carbon_cost: float
    price_cost: float
    notes: List[str] = field(default_factory=list)

    def add_note(self, message: str) -> None:
        self.notes.append(message)


@dataclass
class QueryCostResult:
    query: QuerySpec
    total: CostBreakdown
    component_costs: Dict[str, CostBreakdown]
    join_overhead: Optional[CostBreakdown] = None


@dataclass
class AggregateResult:
    """Cost and sizing output for an aggregate query."""

    label: str
    collection: str
    grouping_keys: List[str]
    output_fields: List[str]
    filtered_key: Optional[str]
    output_documents: int
    output_size_gb: float
    map_cost: CostBreakdown
    shuffle_cost: CostBreakdown
    reduce_cost: CostBreakdown
    total_cost: CostBreakdown
    notes: List[str] = field(default_factory=list)
