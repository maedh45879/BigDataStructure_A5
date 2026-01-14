from __future__ import annotations

BYTES_PER_GB = 1024**3

# Cost units (per GB scanned).
BASE_IO_TIME_UNIT = 1.0
BASE_IO_CARBON_UNIT = 0.5
BASE_IO_PRICE_UNIT = 0.1

# Network multiplier for shuffles and remote communication.
NETWORK_MULTIPLIER = 5.0

# Fraction of shards touched when sharding key is targeted.
DEFAULT_SHARDING_ACCESS_FRACTION = 0.1

# Overhead per field in output documents (bytes).
KEY_OVERHEAD_BYTES = 12

# Leaderboard weighted score coefficients.
WEIGHT_TIME = 1.0
WEIGHT_CARBON = 1.0
WEIGHT_PRICE = 1.0
