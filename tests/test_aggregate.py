import unittest
from pathlib import Path

from query_simulator.aggregate import aggregate_with_sharding, aggregate_without_sharding
from query_simulator.config import STATS
from main import CollectionModel, CollectionStats, load_json_schema


BASE_DIR = Path(__file__).resolve().parents[1]


class AggregateQueriesTest(unittest.TestCase):
    def _load_schema(self, name: str) -> dict:
        db1 = load_json_schema(BASE_DIR / "schemas" / "db1.json")
        return db1[name]

    def test_q6_group_by_sharding_key_reduces_shuffle(self) -> None:
        stock_schema = self._load_schema("Stock")
        stock_stats = CollectionStats(
            nb_documents=int(STATS["N_Prod"] * STATS["AVG_STOCK_PER_PROD"]),
            sharding_key_cardinality={"IDP": int(STATS["N_Prod"])},
            field_cardinality={"IDP": int(STATS["N_Prod"])},
            sharding_key="IDP",
        )
        stock = CollectionModel("Stock", stock_schema, stock_stats)

        with_shard = aggregate_with_sharding(
            collection=stock,
            grouping_keys=["IDP"],
            output_fields=["IDP", "total_quantity"],
        )
        without_shard = aggregate_without_sharding(
            collection=stock,
            grouping_keys=["IDP"],
            output_fields=["IDP", "total_quantity"],
        )

        self.assertGreater(with_shard.output_documents, 0)
        self.assertGreaterEqual(with_shard.output_size_gb, 0.0)
        self.assertGreaterEqual(with_shard.total_cost.time_cost, 0.0)
        self.assertLessEqual(
            with_shard.shuffle_cost.data_scanned_gb,
            without_shard.shuffle_cost.data_scanned_gb,
        )

    def test_q7_group_by_date_without_sharding_has_shuffle(self) -> None:
        order_schema = self._load_schema("OrderLine")
        order_stats = CollectionStats(
            nb_documents=int(STATS["N_OL"]),
            sharding_key_cardinality={"IDC": int(STATS["N_Cl"])},
            field_cardinality={"date": 365},
            sharding_key="IDC",
        )
        orders = CollectionModel("OrderLine", order_schema, order_stats)

        result = aggregate_without_sharding(
            collection=orders,
            grouping_keys=["date"],
            output_fields=["date", "total_quantity"],
        )

        self.assertEqual(result.output_documents, 365)
        self.assertGreater(result.shuffle_cost.data_scanned_gb, 0.0)
        self.assertGreaterEqual(result.total_cost.carbon_cost, 0.0)


if __name__ == "__main__":
    unittest.main()
