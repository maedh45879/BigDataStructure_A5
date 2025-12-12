"""
Configuration and sizing constants for the QuerySimulator.

The values below bundle the statistics provided for Homework 2, alongside
simple helpers to derive collection sizes (in GB) for the different database
signatures (DB1–DB5).
"""

from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Base statistics (provided by the assignment)
# ---------------------------------------------------------------------------

STATS: Dict[str, float] = {
    "N_SERVERS": 1000,
    "N_Cl": 10**7,  # Clients
    "N_Prod": 10**5,  # Products
    "N_OL": 4 * 10**9,  # Order Lines
    "N_Wa": 200,  # Warehouses
    "N_Brands": 5000,  # Distinct Brands
    "AVG_CAT_PER_PROD": 2,
    "AVG_STOCK_PER_PROD": 200,  # N_Wa
    # Estimated selectivities for testing Q1-Q5
    "SEL_Q1_STOCK": 1 / (10**5 * 200),  # 1 product in 1 warehouse out of all stock entries
    "SEL_Q2_BRAND": 50 / 10**5,  # 50 Apple products out of 10^5 total products
    "SEL_Q3_DATE": 1 / 365,  # Order lines from 1 date out of 365
}

# Placeholder values for un-denormalized document sizes (Bytes)
DOC_SIZES_BYTES: Dict[str, int] = {
    "Product": 800,
    "Stock": 300,
    "OrderLine": 500,
    "Client": 700,
    "Warehouse": 400,
    "Supplier": 600,
    "Category": 200,
}

# IO and network cost parameters
BASE_IO_TIME_UNIT = 1.0  # Time/GB
BASE_IO_CARBON_UNIT = 0.5  # Carbon/GB
BASE_IO_PRICE_UNIT = 0.1  # Price/GB
NETWORK_MULTIPLIER = 5.0  # Applied to network communication/full-cluster scans
SHARDING_ACCESS_FRACTION = 1 / 10  # Fraction of shards touched for targeted lookups

# Utility constants
BYTES_PER_GB = 1024**3
DB_SIGNATURES = ("DB1", "DB2", "DB3", "DB4", "DB5")
DEFAULT_DB_SIGNATURE = "DB1"


# ---------------------------------------------------------------------------
# Collection sizing helpers
# ---------------------------------------------------------------------------

def bytes_to_gb(value: float) -> float:
    return value / BYTES_PER_GB


def derive_collection_counts() -> Dict[str, int]:
    """Compute document counts from STATS for each base collection."""
    return {
        "Product": int(STATS["N_Prod"]),
        "Stock": int(STATS["N_Prod"] * STATS["AVG_STOCK_PER_PROD"]),
        "OrderLine": int(STATS["N_OL"]),
        "Client": int(STATS["N_Cl"]),
        "Warehouse": int(STATS["N_Wa"]),
        "Supplier": int(STATS.get("N_Suppliers", STATS["N_Wa"])),
        "Category": int(STATS["N_Prod"] * STATS["AVG_CAT_PER_PROD"]),
    }


def build_collection_sizes_gb() -> Dict[str, Dict[str, float]]:
    """
    Build a mapping of collection -> size (GB) for each database signature.

    The same sizing is reused for DB1–DB5 by default, but the structure lets
    you override a specific signature if your Homework 1 calculations differ.
    """
    counts = derive_collection_counts()
    base_sizes = {
        coll: bytes_to_gb(DOC_SIZES_BYTES[coll] * count)
        for coll, count in counts.items()
        if coll in DOC_SIZES_BYTES
    }
    return {sig: dict(base_sizes) for sig in DB_SIGNATURES}


COLLECTION_SIZES_GB: Dict[str, Dict[str, float]] = build_collection_sizes_gb()


def collection_size_gb(collection: str, db_signature: str = DEFAULT_DB_SIGNATURE) -> float:
    """
    Get the size (GB) of a collection for a given database signature.

    Falls back to 0.0 if the signature or collection is unknown to keep the
    simulator usable even with partial inputs.
    """
    return COLLECTION_SIZES_GB.get(db_signature, {}).get(collection, 0.0)
