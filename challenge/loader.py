from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .config import DEFAULT_SHARDING_ACCESS_FRACTION
from .models import (
    ClusterConfig,
    CollectionConfig,
    CollectionModel,
    CollectionSchema,
    CollectionStats,
    DenormalizationSpec,
    EmbedSpec,
    FieldSpec,
)


_DEFAULT_FIELD_SIZES = {
    "integer": 8,
    "number": 8,
    "string": 80,
    "boolean": 8,
}


def _field_size(raw: dict) -> int:
    if "avg_size" in raw:
        return int(raw["avg_size"])
    return int(_DEFAULT_FIELD_SIZES.get(raw.get("type", "string"), 80))


def load_schema(schema_path: Path) -> Dict[str, CollectionSchema]:
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    collections: Dict[str, CollectionSchema] = {}
    for name, raw in payload.get("collections", {}).items():
        fields: Dict[str, FieldSpec] = {}
        for field_name, field_raw in raw.get("fields", {}).items():
            fields[field_name] = FieldSpec(
                name=field_name,
                avg_size=_field_size(field_raw),
            )
        collections[name] = CollectionSchema(
            name=name,
            primary_key=raw["primary_key"],
            fields=fields,
        )
    return collections


def load_stats(stats_path: Path) -> Tuple[ClusterConfig, Dict[str, CollectionStats], Dict[str, float]]:
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    cluster_raw = payload.get("cluster", {})
    cluster = ClusterConfig(
        nb_servers=int(cluster_raw.get("nb_servers", 1000)),
        sharding_access_fraction=float(
            cluster_raw.get("sharding_access_fraction", DEFAULT_SHARDING_ACCESS_FRACTION)
        ),
    )

    stats: Dict[str, CollectionStats] = {}
    for name, raw in payload.get("collections", {}).items():
        stats[name] = CollectionStats(
            nb_documents=int(raw["nb_documents"]),
            distinct_values={k: int(v) for k, v in raw.get("distinct_values", {}).items()},
            avg_array_lengths={
                k: float(v) for k, v in raw.get("avg_array_lengths", {}).items()
            },
            field_selectivity={
                k: float(v) for k, v in raw.get("field_selectivity", {}).items()
            },
        )
    frequencies = {k: float(v) for k, v in payload.get("query_frequencies", {}).items()}
    return cluster, stats, frequencies


def load_denormalizations(denorm_path: Path) -> List[DenormalizationSpec]:
    payload = json.loads(denorm_path.read_text(encoding="utf-8"))
    denorms: List[DenormalizationSpec] = []
    for raw in payload.get("denormalizations", []):
        collections: Dict[str, CollectionConfig] = {}
        for name, config in raw.get("collections", {}).items():
            collections[name] = CollectionConfig(
                sharding_key=config.get("sharding_key", ""),
                indexes=list(config.get("indexes", [])),
            )
        embeds = [
            EmbedSpec(
                source=item["from"],
                target=item["to"],
                path=item["path"],
                cardinality=item.get("cardinality", "one"),
            )
            for item in raw.get("embeds", [])
        ]
        denorms.append(
            DenormalizationSpec(
                id=raw["id"],
                description=raw.get("description", ""),
                collections=collections,
                embeds=embeds,
            )
        )
    return denorms


def _extend_schema_for_embed(
    base_schema: CollectionSchema,
    embed_schema: CollectionSchema,
    path: str,
    cardinality: str,
) -> CollectionSchema:
    fields = dict(base_schema.fields)
    array_path = path if cardinality == "many" else None
    for field_name, spec in embed_schema.fields.items():
        embedded_name = f"{path}.{field_name}"
        fields[embedded_name] = FieldSpec(
            name=embedded_name,
            avg_size=spec.avg_size,
            array_path=array_path,
        )
    return CollectionSchema(
        name=base_schema.name,
        primary_key=base_schema.primary_key,
        fields=fields,
    )


def _extend_stats_for_embed(
    base_stats: CollectionStats,
    embed_stats: CollectionStats,
    path: str,
) -> CollectionStats:
    distinct_values = dict(base_stats.distinct_values)
    field_selectivity = dict(base_stats.field_selectivity)
    for field_name, value in embed_stats.distinct_values.items():
        distinct_values[f"{path}.{field_name}"] = value
    for field_name, value in embed_stats.field_selectivity.items():
        field_selectivity[f"{path}.{field_name}"] = value
    return replace(
        base_stats,
        distinct_values=distinct_values,
        field_selectivity=field_selectivity,
    )


def build_database_models(
    schemas: Dict[str, CollectionSchema],
    stats: Dict[str, CollectionStats],
    denorm: DenormalizationSpec,
) -> Dict[str, CollectionModel]:
    models: Dict[str, CollectionModel] = {}
    for name, config in denorm.collections.items():
        if name not in schemas:
            raise ValueError(f"Unknown collection in denormalization: {name}")
        if name not in stats:
            raise ValueError(f"Missing stats for collection: {name}")
        sharding_key = config.sharding_key or schemas[name].primary_key
        models[name] = CollectionModel(
            schema=schemas[name],
            stats=stats[name],
            config=CollectionConfig(sharding_key=sharding_key, indexes=list(config.indexes)),
        )

    for embed in denorm.embeds:
        if embed.target not in models:
            continue
        if embed.source not in schemas or embed.source not in stats:
            raise ValueError(f"Embed source missing from schema/stats: {embed.source}")
        target = models[embed.target]
        updated_schema = _extend_schema_for_embed(
            target.schema, schemas[embed.source], embed.path, embed.cardinality
        )
        updated_stats = _extend_stats_for_embed(target.stats, stats[embed.source], embed.path)
        models[embed.target] = CollectionModel(
            schema=updated_schema,
            stats=updated_stats,
            config=target.config,
        )
    return models


def load_queries(queries_path: Path) -> List[dict]:
    payload = json.loads(queries_path.read_text(encoding="utf-8"))
    return list(payload.get("queries", []))


def collect_embed_paths(denorm: DenormalizationSpec) -> Dict[Tuple[str, str], EmbedSpec]:
    return {(embed.source, embed.target): embed for embed in denorm.embeds}
