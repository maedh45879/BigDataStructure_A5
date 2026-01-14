from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class FieldSpec:
    name: str
    avg_size: int
    array_path: Optional[str] = None


@dataclass
class CollectionSchema:
    name: str
    primary_key: str
    fields: Dict[str, FieldSpec]

    def field_size_bytes(self, field_path: str, avg_array_lengths: Dict[str, float]) -> int:
        spec = self.fields.get(field_path)
        if spec is None:
            return 0
        multiplier = 1.0
        if spec.array_path:
            multiplier = avg_array_lengths.get(spec.array_path, 1.0)
        return int(spec.avg_size * multiplier)

    def document_size_bytes(self, avg_array_lengths: Dict[str, float]) -> int:
        total = 0
        for spec in self.fields.values():
            multiplier = 1.0
            if spec.array_path:
                multiplier = avg_array_lengths.get(spec.array_path, 1.0)
            total += int(spec.avg_size * multiplier)
        return total


@dataclass
class CollectionStats:
    nb_documents: int
    distinct_values: Dict[str, int] = field(default_factory=dict)
    avg_array_lengths: Dict[str, float] = field(default_factory=dict)
    field_selectivity: Dict[str, float] = field(default_factory=dict)


@dataclass
class CollectionConfig:
    sharding_key: str
    indexes: List[str] = field(default_factory=list)


@dataclass
class CollectionModel:
    schema: CollectionSchema
    stats: CollectionStats
    config: CollectionConfig

    def document_size_bytes(self) -> int:
        return self.schema.document_size_bytes(self.stats.avg_array_lengths)


@dataclass
class ClusterConfig:
    nb_servers: int
    sharding_access_fraction: float


@dataclass
class EmbedSpec:
    source: str
    target: str
    path: str
    cardinality: str


@dataclass
class DenormalizationSpec:
    id: str
    description: str
    collections: Dict[str, CollectionConfig]
    embeds: List[EmbedSpec]


@dataclass
class QuerySpec:
    id: str
    sql: str
    frequency: float = 1.0


@dataclass
class FilterPredicate:
    collection: str
    field: str
    value: object


@dataclass
class JoinPredicate:
    left_collection: str
    left_field: str
    right_collection: str
    right_field: str


@dataclass
class PlanOperator:
    name: str
    operator_type: str
    target_collection: Optional[str] = None
    left_collection: Optional[str] = None
    right_collection: Optional[str] = None
    filters: List[FilterPredicate] = field(default_factory=list)
    join: Optional[JoinPredicate] = None
    grouping_keys: List[str] = field(default_factory=list)
    output_fields: List[str] = field(default_factory=list)
    scan_strategy: Optional[str] = None
    indexes_used: List[str] = field(default_factory=list)
    use_sharding: bool = True


@dataclass
class QueryPlan:
    query: QuerySpec
    operators: List[PlanOperator]
    involved_collections: List[str]


@dataclass
class CostBreakdown:
    label: str
    data_scanned_gb: float
    time_cost: float
    carbon_cost: float
    price_cost: float
    notes: List[str] = field(default_factory=list)


@dataclass
class OperatorMetrics:
    operator: PlanOperator
    scanned_docs: int
    output_docs: int
    scanned_bytes: int
    output_bytes: int
    shuffled_bytes: int
    output_doc_size_bytes: int
    cost: CostBreakdown
    details: List[CostBreakdown] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class QueryResult:
    plan: QueryPlan
    operators: List[OperatorMetrics]
    total_cost: CostBreakdown
    scanned_docs: int
    output_docs: int
    scanned_bytes: int
    output_bytes: int
    shuffled_bytes: int


def plan_summary(operators: Sequence[PlanOperator]) -> str:
    return " -> ".join(op.operator_type for op in operators)
