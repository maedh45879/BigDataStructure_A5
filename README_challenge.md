# Chapter 5 Challenge

This module evaluates query costs across denormalization signatures using a
simple operator pipeline (filter/join/aggregate) and a configurable cost model.

## Input formats

### schema.json
Defines the logical collections and per-field average sizes (bytes).

```json
{
  "collections": {
    "Product": {
      "primary_key": "IDP",
      "fields": {
        "IDP": { "type": "integer", "avg_size": 8 },
        "brand": { "type": "string", "avg_size": 40 }
      }
    }
  }
}
```

### stats.json
Provides counts, distinct values, and cluster settings for selectivity and sharding.

```json
{
  "cluster": { "nb_servers": 1000, "sharding_access_fraction": 0.1 },
  "collections": {
    "Product": {
      "nb_documents": 100000,
      "distinct_values": { "IDP": 100000, "brand": 5000 },
      "avg_array_lengths": { "orderLines": 40 }
    }
  },
  "query_frequencies": { "Q1": 1000 }
}
```

### denormalizations.json
Each signature lists the collections to keep, their sharding key and indexes,
and optional embeddings.

```json
{
  "denormalizations": [
    {
      "id": "D1",
      "description": "Normalized",
      "collections": {
        "Product": { "sharding_key": "IDP", "indexes": ["brand"] }
      },
      "embeds": []
    },
    {
      "id": "D2",
      "description": "OrderLine embeds Product",
      "collections": {
        "OrderLine": { "sharding_key": "IDP", "indexes": ["IDC", "product.brand"] }
      },
      "embeds": [
        { "from": "Product", "to": "OrderLine", "path": "product", "cardinality": "one" }
      ]
    }
  ]
}
```

### queries.json
Supports:
- `SELECT fields FROM Collection WHERE field = value`
- `SELECT fields FROM A JOIN B ON A.x = B.y WHERE predicates`

```json
{
  "queries": [
    { "id": "Q1", "sql": "SELECT description FROM Product WHERE categorie = 'smartphone';", "frequency": 1000 }
  ]
}
```

## How denormalization works
- Embedding adds `path.field` to the target collection.
- `cardinality: "many"` multiplies embedded fields by `avg_array_lengths[path]`.
- Joins are removed when all referenced fields are available through an embed.

## Run the CLI

```bash
python -m challenge.run --schema challenge/schema.json --stats challenge/stats.json --denorm challenge/denormalizations.json --queries challenge/queries.json --out out/
```

Outputs:
- `out/plans/q1.plan.json`, `out/plans/q2.plan.json` (plans per denormalization)
- `out/results.csv` (per query results)
- `out/leaderboard.md` (ranked denormalizations)

## Result interpretation
- `scanned_docs` / `scanned_bytes`: estimated IO volume.
- `returned_bytes`: final query output size.
- Costs include time, carbon, and price. `leaderboard.md` also provides a weighted score
  (configured in `challenge/config.py`).
