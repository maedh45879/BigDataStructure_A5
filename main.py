from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# 1) Loading JSON Schemas
# ---------------------------------------------------------------------------

def load_json_schema(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 2) Statistics structures
# ---------------------------------------------------------------------------

@dataclass
class CollectionStats:
    # number of documents in the collection
    nb_documents: int
    # avg length of arrays, by "path" name (ex: "categories", "stocks", "orderLines")
    avg_array_lengths: Dict[str, int] = field(default_factory=dict)
    # cardinality of sharding keys (ex: {"IDP": 100000, "brand": 5000})
    sharding_key_cardinality: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3) Size computation helpers
# ---------------------------------------------------------------------------

def primitive_value_size(json_type: str, format_: Optional[str] = None) -> int:
    json_type = json_type.lower()
    if json_type in {"integer", "number"}:
        return 8            # Integer/Number : 8B
    if json_type == "string":
        if format_ == "date":
            return 20       # Date : 20B
        return 80           # String : 80B
    if json_type in {"boolean", "null"}:
        return 8
    return 0


def compute_document_size_bytes(
    schema: Dict[str, Any],
    stats: Optional[CollectionStats] = None,
    prefix: str = ""
) -> int:
    """
    Recursively compute the size of one document of this schema, in bytes.
    prefix is used to build a "path" name for arrays (ex: "categories").
    """
    if stats is None:
        stats = CollectionStats(nb_documents=1)

    total = 0

    # object
    if schema.get("type") == "object":
        props = schema.get("properties", {})
        for prop_name, prop_schema in props.items():
            full_name = f"{prefix}{prop_name}"
            # key + value overhead
            total += 12
            # value
            total += compute_document_size_bytes(
                prop_schema, stats, prefix=f"{full_name}."
            )

    # array
    elif schema.get("type") == "array":
        items_schema = schema.get("items", {})
        key = prefix.rstrip(".")
        avg_len = stats.avg_array_lengths.get(key, 1)
        element_size = compute_document_size_bytes(
            items_schema, stats, prefix=prefix
        )
        total += avg_len * element_size

    # primitive
    else:
        json_type = schema.get("type", "string")
        format_ = schema.get("format")
        total += primitive_value_size(json_type, format_)

    return total


def compute_collection_size_gb(
    schema: Dict[str, Any],
    stats: CollectionStats
) -> float:
    doc_size = compute_document_size_bytes(schema, stats)
    total_bytes = doc_size * stats.nb_documents
    return total_bytes / (1024 ** 3)


# ---------------------------------------------------------------------------
# 4) Database model
# ---------------------------------------------------------------------------

@dataclass
class CollectionModel:
    name: str
    schema: Dict[str, Any]
    stats: CollectionStats

    def document_size_bytes(self) -> int:
        return compute_document_size_bytes(self.schema, self.stats)

    def size_gb(self) -> float:
        return compute_collection_size_gb(self.schema, self.stats)


@dataclass
class DatabaseModel:
    collections: Dict[str, CollectionModel] = field(default_factory=dict)

    def add_collection(self, coll: CollectionModel) -> None:
        self.collections[coll.name] = coll

    def total_size_gb(self) -> float:
        return sum(c.size_gb() for c in self.collections.values())


# ---------------------------------------------------------------------------
# 5) Sharding statistics
# ---------------------------------------------------------------------------

def compute_sharding_stats(
    coll: CollectionModel,
    sharding_key: str,
    nb_servers: int = 1000
) -> Dict[str, float]:
    """
    Returns:
      - avg_docs_per_server
      - avg_distinct_key_values_per_server
    """
    nb_docs = coll.stats.nb_documents
    avg_docs_per_server = nb_docs / nb_servers

    key_card = coll.stats.sharding_key_cardinality.get(sharding_key)
    if key_card is None:
        avg_distinct_per_server = float("nan")
    else:
        avg_distinct_per_server = key_card / nb_servers

    return {
        "collection": coll.name,
        "sharding_key": sharding_key,
        "nb_servers": nb_servers,
        "avg_docs_per_server": avg_docs_per_server,
        "avg_distinct_key_values_per_server": avg_distinct_per_server,
    }


# ---------------------------------------------------------------------------
# 6) Example usage (you can adapt this part for your JSON Schemas DB1..DB5)
# ---------------------------------------------------------------------------

def example() -> None:
    # Example: load a Product JSON schema from schemas/product.json
    product_schema_path = Path("schemas/product.json")
    if not product_schema_path.exists():
        print("schemas/product.json not found, skip example.")
        return

    product_schema = load_json_schema(product_schema_path)

    # Adapt these stats with the real numbers from the subject
    product_stats = CollectionStats(
        nb_documents=10**5,
        avg_array_lengths={
            # example: path "categories" (if your schema has "categories": { "type": "array", ... })
            "categories": 2,
        },
        sharding_key_cardinality={
            "IDP": 10**5,
            "brand": 5000,
        },
    )

    product_collection = CollectionModel(
        name="Prod",
        schema=product_schema,
        stats=product_stats,
    )

    db = DatabaseModel()
    db.add_collection(product_collection)

    print("Document size (bytes):", product_collection.document_size_bytes())
    print("Collection size (GB):", product_collection.size_gb())
    print("Database total size (GB):", db.total_size_gb())

    for key in ["IDP", "brand"]:
        stats = compute_sharding_stats(product_collection, key, nb_servers=1000)
        print(stats)


if __name__ == "__main__":
    example()
