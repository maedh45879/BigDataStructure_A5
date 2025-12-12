from __future__ import annotations

from typing import List

from .config import STATS
from .models import QueryComponent, QuerySpec


def default_queries() -> List[QuerySpec]:
    """
    Default Q1–Q5 queries used in Homework 2 section 3.3.

    The selectivities are aligned with the STATS constants and can be edited
    to reflect updated assumptions.
    """
    q1 = QuerySpec(
        name="Q1_stock_lookup",
        kind="filter",
        description="Check stock for one product in one warehouse.",
        components=[
            QueryComponent(
                name="Stock filter",
                collection="Stock",
                selectivity=STATS["SEL_Q1_STOCK"],
                description="Product + warehouse filter on stock",
                targeted_shard=True,
                requires_network=False,
            )
        ],
    )

    q2 = QuerySpec(
        name="Q2_brand_filter",
        kind="filter",
        description="Find Apple products by brand.",
        components=[
            QueryComponent(
                name="Product brand filter",
                collection="Product",
                selectivity=STATS["SEL_Q2_BRAND"],
                description="Brand = Apple (50 products)",
                targeted_shard=False,
                requires_network=True,
            )
        ],
    )

    q3 = QuerySpec(
        name="Q3_orders_by_date",
        kind="filter",
        description="Fetch order lines placed on a specific date.",
        components=[
            QueryComponent(
                name="OrderLine date filter",
                collection="OrderLine",
                selectivity=STATS["SEL_Q3_DATE"],
                description="Order date = target day",
                targeted_shard=False,
                requires_network=True,
            )
        ],
    )

    q4 = QuerySpec(
        name="Q4_brand_stock_join",
        kind="join",
        description="Join Apple products with their per-warehouse stock.",
        components=[
            QueryComponent(
                name="Product brand filter",
                collection="Product",
                selectivity=STATS["SEL_Q2_BRAND"],
                description="Brand = Apple",
                targeted_shard=False,
                requires_network=True,
            ),
            QueryComponent(
                name="Stock by product",
                collection="Stock",
                selectivity=STATS["SEL_Q2_BRAND"],
                description="Stock entries for Apple products",
                targeted_shard=True,
                requires_network=False,
            ),
        ],
        join_selectivity=STATS["SEL_Q2_BRAND"],
    )

    q5 = QuerySpec(
        name="Q5_orders_brand_client_join",
        kind="join",
        description=(
            "Orders on a target date joined with Apple products and client info "
            "for downstream personalization/analytics."
        ),
        components=[
            QueryComponent(
                name="OrderLine date filter",
                collection="OrderLine",
                selectivity=STATS["SEL_Q3_DATE"],
                description="Order date = target day",
                targeted_shard=False,
                requires_network=True,
            ),
            QueryComponent(
                name="Product brand filter",
                collection="Product",
                selectivity=STATS["SEL_Q2_BRAND"],
                description="Brand = Apple",
                targeted_shard=False,
                requires_network=True,
            ),
            QueryComponent(
                name="Client lookup",
                collection="Client",
                selectivity=1 / STATS["N_Cl"],
                description="Join to one client record per matching order",
                targeted_shard=True,
                requires_network=False,
            ),
        ],
        join_selectivity=STATS["SEL_Q2_BRAND"] * STATS["SEL_Q3_DATE"],
    )

    return [q1, q2, q3, q4, q5]
