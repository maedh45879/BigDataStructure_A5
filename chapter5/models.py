from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from query_simulator.models import CostBreakdown


OperatorType = Literal["filter", "join", "aggregate"]


@dataclass
class PlanQuerySpec:
    name: str
    description: str
    frequency: float = 1.0


@dataclass
class PlanStep:
    name: str
    operator_type: OperatorType
    target_collection: Optional[str] = None
    input_ref: Optional[str] = None
    left_ref: Optional[str] = None
    right_ref: Optional[str] = None
    filter_key: Optional[str] = None
    selectivity: Optional[float] = None
    join_key: Optional[str] = None
    join_selectivity: Optional[float] = None
    grouping_keys: List[str] = field(default_factory=list)
    output_fields: List[str] = field(default_factory=list)
    use_sharding: bool = True


@dataclass
class QueryPlan:
    query: PlanQuerySpec
    steps: List[PlanStep]


@dataclass
class StepResult:
    step: PlanStep
    output_documents: int
    output_doc_size_bytes: int
    output_size_gb: float
    cost: CostBreakdown
    details: List[CostBreakdown] = field(default_factory=list)
    field_sizes: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    sharding_key: Optional[str] = None


@dataclass
class PlanResult:
    plan: QueryPlan
    steps: List[StepResult]
    total_cost: CostBreakdown
    output_documents: int
    output_size_gb: float
