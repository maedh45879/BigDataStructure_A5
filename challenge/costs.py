from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from .config import BASE_IO_CARBON_UNIT, BASE_IO_PRICE_UNIT, BASE_IO_TIME_UNIT, NETWORK_MULTIPLIER
from .models import CostBreakdown


@dataclass
class CostModel:
    base_time_unit: float = BASE_IO_TIME_UNIT
    base_carbon_unit: float = BASE_IO_CARBON_UNIT
    base_price_unit: float = BASE_IO_PRICE_UNIT
    network_multiplier: float = NETWORK_MULTIPLIER

    def io_cost(self, label: str, data_gb: float, use_network: bool = False) -> CostBreakdown:
        multiplier = self.network_multiplier if use_network else 1.0
        return CostBreakdown(
            label=label,
            data_scanned_gb=data_gb,
            time_cost=data_gb * self.base_time_unit * multiplier,
            carbon_cost=data_gb * self.base_carbon_unit * multiplier,
            price_cost=data_gb * self.base_price_unit * multiplier,
            notes=[f"Network multiplier x{self.network_multiplier}"] if use_network else [],
        )

    def aggregate(self, label: str, parts: Iterable[CostBreakdown]) -> CostBreakdown:
        parts_list = list(parts)
        return CostBreakdown(
            label=label,
            data_scanned_gb=sum(part.data_scanned_gb for part in parts_list),
            time_cost=sum(part.time_cost for part in parts_list),
            carbon_cost=sum(part.carbon_cost for part in parts_list),
            price_cost=sum(part.price_cost for part in parts_list),
            notes=[note for part in parts_list for note in part.notes],
        )
