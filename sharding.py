from collections import defaultdict
import hashlib

def _hash_slot(value: str, n: int) -> int:
    h = int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)
    return h % n

def simulate_sharding_counts(num_docs: int, distinct_keys: int, n_servers: int,
                             strategy: str = "hash", replication: int = 1):
    if n_servers <= 0:
        raise ValueError("Number of servers must be > 0")

    distinct_keys = max(1, int(distinct_keys))
    num_docs = max(0, int(num_docs))
    docs_per_key = num_docs // distinct_keys

    keys_per_server = defaultdict(int)

    if strategy == "hash":
        for k in range(distinct_keys):
            srv = _hash_slot(str(k), n_servers)
            keys_per_server[srv] += 1
    elif strategy == "range":
        base = distinct_keys // n_servers
        rem = distinct_keys % n_servers
        for s in range(n_servers):
            keys_per_server[s] = base + (1 if s < rem else 0)
    else:
        raise ValueError("Strategy must be 'hash' or 'range'")

    docs_per_server = {s: keys_per_server[s] * docs_per_key for s in range(n_servers)}
    distinct_per_server = dict(keys_per_server)

    return {
        "docs_per_server": docs_per_server,
        "distinct_keys_per_server": distinct_per_server,
        "avg_docs_per_server": num_docs / n_servers,
        "avg_distinct_keys_per_server": distinct_keys / n_servers,
        "replication_factor": replication,
        "storage_multiplier": replication
    }
