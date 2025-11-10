import json
import argparse
from sharding import simulate_sharding_counts

def main():
    parser = argparse.ArgumentParser(description="Person 3 – Sharding Simulation")
    parser.add_argument("--stats_file", default="stats.json")
    parser.add_argument("--servers", type=int, default=1000)
    parser.add_argument("--strategy", choices=["hash", "range"], default="hash")
    parser.add_argument("--replication", type=int, default=1)
    args = parser.parse_args()

    with open(args.stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)

    print("\n== SHARDING SIMULATION ==")
    for coll in stats["collections"]:
        name = coll["name"]
        num_docs = coll["nb_documents"]

        # run once for each available sharding key in that collection
        for key, distinct in coll.get("sharding_key_cardinality", {}).items():
            out = simulate_sharding_counts(
                num_docs=num_docs,
                distinct_keys=distinct,
                n_servers=args.servers,
                strategy=args.strategy,
                replication=args.replication,
            )
            print(f"\nCollection: {name}  | Key: {key}")
            print(f"Strategy: {args.strategy} | Servers: {args.servers}")
            print(f"Avg docs/server: {out['avg_docs_per_server']:.2f}")
            print(f"Avg distinct/server: {out['avg_distinct_keys_per_server']:.2f}")
            print(f"Replication factor: {args.replication} (storage x{out['storage_multiplier']})")

if __name__ == "__main__":
    main()
