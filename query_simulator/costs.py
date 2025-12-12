from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .config import (
    BASE_IO_CARBON_UNIT,
    BASE_IO_PRICE_UNIT,
    BASE_IO_TIME_UNIT,
    DEFAULT_DB_SIGNATURE,
    NETWORK_MULTIPLIER,
    SHARDING_ACCESS_FRACTION,
    collection_size_gb,
)
from .models import CostBreakdown, QueryComponent, QueryCostResult, QuerySpec


@dataclass
class QueryCostModel:
    """Cost model applying IO-based time/carbon/price units."""

    base_time_unit: float = BASE_IO_TIME_UNIT
    base_carbon_unit: float = BASE_IO_CARBON_UNIT
    base_price_unit: float = BASE_IO_PRICE_UNIT
    network_multiplier: float = NETWORK_MULTIPLIER
    sharding_access_fraction: float = SHARDING_ACCESS_FRACTION

    def io_cost(
        self,
        label: str,
        data_gb: float,
        use_network: bool = False,
        notes: Optional[List[str]] = None,
    ) -> CostBreakdown:
        multiplier = self.network_multiplier if use_network else 1.0
        note_list = list(notes or [])
        if use_network and self.network_multiplier != 1.0:
            note_list.append(f"Network multiplier x{self.network_multiplier}")
        return CostBreakdown(
            label=label,
            data_scanned_gb=data_gb,
            time_cost=data_gb * self.base_time_unit * multiplier,
            carbon_cost=data_gb * self.base_carbon_unit * multiplier,
            price_cost=data_gb * self.base_price_unit * multiplier,
            notes=note_list,
        )

    def aggregate(self, label: str, parts: Iterable[CostBreakdown]) -> CostBreakdown:
        parts_list = list(parts)
        return CostBreakdown(
            label=label,
            data_scanned_gb=sum(p.data_scanned_gb for p in parts_list),
            time_cost=sum(p.time_cost for p in parts_list),
            carbon_cost=sum(p.carbon_cost for p in parts_list),
            price_cost=sum(p.price_cost for p in parts_list),
            notes=[note for p in parts_list for note in p.notes],
        )

    def estimate_component(self, component: QueryComponent, db_signature: str) -> CostBreakdown:
        base_size_gb = collection_size_gb(component.collection, db_signature)
        data_gb = base_size_gb * component.selectivity

        notes: List[str] = [component.description]

        if component.targeted_shard:
            data_gb *= self.sharding_access_fraction
            notes.append(
                f"Targeted shard access: x{self.sharding_access_fraction} of the cluster"
            )

        if base_size_gb == 0:
            notes.append("Warning: collection size not found, using 0 GB")

        cost = self.io_cost(
            label=component.name,
            data_gb=data_gb,
            use_network=component.requires_network,
            notes=notes,
        )
        return cost

    def estimate_query(
        self,
        query: QuerySpec,
        db_signature: str = DEFAULT_DB_SIGNATURE,
    ) -> QueryCostResult:
        component_costs = {
            comp.name: self.estimate_component(comp, db_signature) for comp in query.components
        }

        if query.kind == "filter":
            total = self.aggregate(f"{query.name}:total", component_costs.values())
            return QueryCostResult(
                query=query,
                total=total,
                component_costs=component_costs,
            )

        if query.kind == "join":
            base_total = self.aggregate(f"{query.name}:components", component_costs.values())
            join_overhead = self.io_cost(
                label=f"{query.name}:join_network",
                data_gb=base_total.data_scanned_gb,
                use_network=True,
                notes=["Network shuffle for join"],
            )
            grand_total = self.aggregate(
                f"{query.name}:total", [base_total, join_overhead]
            )
            if query.join_selectivity is not None:
                grand_total.add_note(
                    f"Join selectivity (result fraction): {query.join_selectivity:.4f}"
                )
            return QueryCostResult(
                query=query,
                total=grand_total,
                component_costs=component_costs,
                join_overhead=join_overhead,
            )

        raise ValueError(f"Unsupported query kind: {query.kind}")
