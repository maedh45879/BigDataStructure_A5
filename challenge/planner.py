from __future__ import annotations

import re
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from .models import (
    FilterPredicate,
    JoinPredicate,
    PlanOperator,
    QueryPlan,
    QuerySpec,
)


_RE_WHITESPACE = re.compile(r"\s+")


def _normalize_sql(sql: str) -> str:
    cleaned = sql.strip().rstrip(";")
    cleaned = _RE_WHITESPACE.sub(" ", cleaned)
    return cleaned


def _parse_select_fields(segment: str) -> List[str]:
    return [field.strip() for field in segment.split(",") if field.strip()]


def _parse_where(where_clause: str, alias_map: Dict[str, str]) -> List[FilterPredicate]:
    predicates: List[FilterPredicate] = []
    for raw in re.split(r"\s+AND\s+", where_clause, flags=re.IGNORECASE):
        match = re.match(r"(?:(\w+)\.)?(\w+)\s*=\s*(.+)$", raw.strip())
        if not match:
            continue
        alias, field, value_raw = match.groups()
        value_raw = value_raw.strip()
        if value_raw.startswith("'") and value_raw.endswith("'"):
            value: object = value_raw.strip("'")
        else:
            try:
                value = int(value_raw)
            except ValueError:
                try:
                    value = float(value_raw)
                except ValueError:
                    value = value_raw
        collection = alias_map.get(alias, alias_map.get("", ""))
        predicates.append(FilterPredicate(collection=collection, field=field, value=value))
    return predicates


def _parse_join(sql: str) -> Optional[Dict[str, str]]:
    join_match = re.search(
        r"\sJOIN\s+(?P<right>\w+)(?:\s+(?P<right_alias>\w+))?\s+ON\s+(?P<left_alias>\w+)\.(?P<left_key>\w+)\s*=\s*(?P<right_alias2>\w+)\.(?P<right_key>\w+)",
        sql,
        flags=re.IGNORECASE,
    )
    if not join_match:
        return None
    data = join_match.groupdict()
    return {
        "right": data["right"],
        "right_alias": data.get("right_alias") or data["right"],
        "left_alias": data["left_alias"],
        "left_key": data["left_key"],
        "right_alias2": data["right_alias2"],
        "right_key": data["right_key"],
    }


def parse_sql(sql: str) -> Tuple[List[str], Dict[str, str], Optional[JoinPredicate], List[FilterPredicate]]:
    normalized = _normalize_sql(sql)
    select_match = re.match(r"SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<from>.+)", normalized, flags=re.IGNORECASE)
    if not select_match:
        raise ValueError(f"Unsupported SQL: {sql}")
    select_segment = select_match.group("select")
    remainder = select_match.group("from")
    select_fields = _parse_select_fields(select_segment)

    where_clause = ""
    where_match = re.search(r"\sWHERE\s+(.+)$", remainder, flags=re.IGNORECASE)
    if where_match:
        where_clause = where_match.group(1)
        remainder = remainder[: where_match.start()]

    join_info = _parse_join(remainder)
    if join_info:
        left_part = remainder.split("JOIN", 1)[0].strip()
        left_tokens = left_part.split()
        left_collection = left_tokens[0]
        left_alias = left_tokens[1] if len(left_tokens) > 1 else left_collection
        right_collection = join_info["right"]
        right_alias = join_info["right_alias"]
        alias_map = {
            left_alias: left_collection,
            right_alias: right_collection,
        }
        join_predicate = JoinPredicate(
            left_collection=left_collection,
            left_field=join_info["left_key"],
            right_collection=right_collection,
            right_field=join_info["right_key"],
        )
    else:
        tokens = remainder.strip().split()
        collection = tokens[0]
        alias = tokens[1] if len(tokens) > 1 else ""
        alias_map = {alias: collection, "": collection}
        join_predicate = None

    filters = _parse_where(where_clause, alias_map) if where_clause else []
    return select_fields, alias_map, join_predicate, filters


def _choose_scan_strategy(
    filter_field: Optional[str],
    sharding_key: str,
    indexes: List[str],
) -> str:
    if filter_field and filter_field == sharding_key:
        return "shard"
    if filter_field and filter_field in indexes:
        return "index"
    return "full"


def _operator_type_for_filter(strategy: str) -> str:
    if strategy == "shard":
        return "filter_with_sharding"
    return "filter_without_sharding"


def _operator_type_for_join(aligned: bool) -> str:
    return "nested_loop_with_sharding" if aligned else "nested_loop_without_sharding"


def _normalize_select_fields(select_fields: List[str], alias_map: Dict[str, str]) -> List[str]:
    normalized: List[str] = []
    for field in select_fields:
        if "." in field:
            alias, name = field.split(".", 1)
            collection = alias_map.get(alias, "")
            if collection:
                normalized.append(f"{collection}.{name}")
            else:
                normalized.append(name)
        else:
            normalized.append(field)
    return normalized


def plan_query(
    query: QuerySpec,
    collections: Dict[str, object],
    embeds: Dict[Tuple[str, str], object],
) -> QueryPlan:
    select_fields, alias_map, join_predicate, filters = parse_sql(query.sql)
    select_fields = _normalize_select_fields(select_fields, alias_map)

    involved_collections = sorted({pred.collection for pred in filters if pred.collection})
    if join_predicate:
        involved_collections = sorted(
            {join_predicate.left_collection, join_predicate.right_collection}
        )

    operators: List[PlanOperator] = []

    if join_predicate:
        embed_left = embeds.get((join_predicate.left_collection, join_predicate.right_collection))
        embed_right = embeds.get((join_predicate.right_collection, join_predicate.left_collection))

        if embed_left or embed_right:
            if embed_left:
                base_collection = join_predicate.right_collection
                prefix = embed_left.path
                prefix_owner = join_predicate.left_collection
            else:
                base_collection = join_predicate.left_collection
                prefix = embed_right.path
                prefix_owner = join_predicate.right_collection

            rewritten_filters: List[FilterPredicate] = []
            for predicate in filters:
                field_name = predicate.field
                if predicate.collection == prefix_owner:
                    field_name = f"{prefix}.{predicate.field}"
                rewritten_filters.append(
                    replace(predicate, collection=base_collection, field=field_name)
                )

            rewritten_select: List[str] = []
            for field in select_fields:
                if field.startswith(f"{prefix_owner}."):
                    _, name = field.split(".", 1)
                    rewritten_select.append(f"{prefix}.{name}")
                elif "." in field:
                    _, name = field.split(".", 1)
                    rewritten_select.append(name)
                else:
                    rewritten_select.append(field)

            config = collections[base_collection].config
            filter_field = rewritten_filters[0].field if rewritten_filters else None
            strategy = _choose_scan_strategy(
                filter_field=filter_field,
                sharding_key=config.sharding_key,
                indexes=config.indexes,
            )
            operators.append(
                PlanOperator(
                    name=f"{query.id}_filter",
                    operator_type=_operator_type_for_filter(strategy),
                    target_collection=base_collection,
                    filters=rewritten_filters,
                    output_fields=rewritten_select,
                    scan_strategy=strategy,
                    indexes_used=[filter_field] if strategy == "index" and filter_field else [],
                    use_sharding=strategy == "shard",
                )
            )
            involved_collections = [base_collection]
            return QueryPlan(query=query, operators=operators, involved_collections=involved_collections)

        left_filters = [pred for pred in filters if pred.collection == join_predicate.left_collection]
        right_filters = [pred for pred in filters if pred.collection == join_predicate.right_collection]

        for name, collection_filters in (
            (join_predicate.left_collection, left_filters),
            (join_predicate.right_collection, right_filters),
        ):
            if not collection_filters:
                continue
            config = collections[name].config
            filter_field = collection_filters[0].field
            strategy = _choose_scan_strategy(
                filter_field=filter_field,
                sharding_key=config.sharding_key,
                indexes=config.indexes,
            )
            operators.append(
                PlanOperator(
                    name=f"{query.id}_filter_{name}",
                    operator_type=_operator_type_for_filter(strategy),
                    target_collection=name,
                    filters=collection_filters,
                    scan_strategy=strategy,
                    indexes_used=[filter_field] if strategy == "index" else [],
                    use_sharding=strategy == "shard",
                )
            )

        left_config = collections[join_predicate.left_collection].config
        right_config = collections[join_predicate.right_collection].config
        aligned = (
            join_predicate.left_field == left_config.sharding_key
            and join_predicate.right_field == right_config.sharding_key
        )
        operators.append(
            PlanOperator(
                name=f"{query.id}_join",
                operator_type=_operator_type_for_join(aligned),
                left_collection=join_predicate.left_collection,
                right_collection=join_predicate.right_collection,
                join=join_predicate,
                output_fields=select_fields,
                use_sharding=aligned,
            )
        )
        return QueryPlan(query=query, operators=operators, involved_collections=involved_collections)

    collection_name = involved_collections[0] if involved_collections else list(alias_map.values())[0]
    if collection_name not in collections:
        for (source, target), embed in embeds.items():
            if source == collection_name and target in collections:
                collection_name = target
                filters = [
                    replace(pred, collection=collection_name, field=f"{embed.path}.{pred.field}")
                    for pred in filters
                ]
                rewritten_select: List[str] = []
                for field in select_fields:
                    if field.startswith(f"{source}."):
                        _, name = field.split(".", 1)
                        rewritten_select.append(f"{embed.path}.{name}")
                    elif "." in field:
                        _, name = field.split(".", 1)
                        rewritten_select.append(name)
                    else:
                        rewritten_select.append(field)
                select_fields = rewritten_select
                break
    config = collections[collection_name].config
    filter_field = filters[0].field if filters else None
    strategy = _choose_scan_strategy(
        filter_field=filter_field,
        sharding_key=config.sharding_key,
        indexes=config.indexes,
    )
    operators.append(
        PlanOperator(
            name=f"{query.id}_filter",
            operator_type=_operator_type_for_filter(strategy),
            target_collection=collection_name,
            filters=filters,
            output_fields=[field.split(".", 1)[-1] for field in select_fields],
            scan_strategy=strategy,
            indexes_used=[filter_field] if strategy == "index" and filter_field else [],
            use_sharding=strategy == "shard",
        )
    )
    return QueryPlan(query=query, operators=operators, involved_collections=[collection_name])
