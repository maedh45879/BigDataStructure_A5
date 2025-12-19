from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from main import CollectionModel, CollectionStats, DatabaseModel, load_json_schema


@dataclass
class ClusterConfig:
    nb_servers: int = 1000
    sharding_access_fraction: float = 0.1


def _resolve_schema_entry(base_dir: Path, schema: dict) -> dict:
    ref = schema.get("$ref")
    if not ref:
        return schema
    return load_json_schema(base_dir / ref)


def load_schema_set(schema_dir: Path, db_name: str) -> Dict[str, dict]:
    schema_path = schema_dir / f"{db_name}.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    raw = load_json_schema(schema_path)
    return {name: _resolve_schema_entry(schema_dir, schema) for name, schema in raw.items()}


def load_stats(stats_path: Path) -> Tuple[ClusterConfig, Dict[str, CollectionStats]]:
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    with stats_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    cluster = payload.get("cluster", {})
    config = ClusterConfig(
        nb_servers=int(cluster.get("nb_servers", 1000)),
        sharding_access_fraction=float(cluster.get("sharding_access_fraction", 0.1)),
    )

    stats_map: Dict[str, CollectionStats] = {}
    for name, raw in payload.get("collections", {}).items():
        stats_map[name] = CollectionStats(
            nb_documents=int(raw["nb_documents"]),
            avg_array_lengths=dict(raw.get("avg_array_lengths", {})),
            sharding_key_cardinality=dict(raw.get("sharding_key_cardinality", {})),
            field_cardinality=dict(raw.get("field_cardinality", {})),
            field_selectivity=dict(raw.get("field_selectivity", {})),
            sharding_key=raw.get("sharding_key"),
        )
    return config, stats_map


def build_database_model(
    schema_dir: Path,
    db_name: str,
    stats_map: Dict[str, CollectionStats],
) -> DatabaseModel:
    db = DatabaseModel()
    schema_set = load_schema_set(schema_dir, db_name)
    for collection_name, schema in schema_set.items():
        if collection_name not in stats_map:
            raise ValueError(f"Missing stats for collection: {collection_name}")
        db.add_collection(CollectionModel(collection_name, schema, stats_map[collection_name]))
    return db


def load_queries(queries_path: Path) -> List[dict]:
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    with queries_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return list(payload.get("queries", []))


def resolve_schema_root(data_root: Path, schema_set: Optional[str]) -> Path:
    if schema_set and schema_set != "default":
        return data_root / schema_set
    return data_root
