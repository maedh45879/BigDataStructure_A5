# Data Model Selection Report

## Executive Summary
- This report consolidates the data-model selection work for filter, join, and aggregate workloads.
- The codebase implements a lightweight simulator to compare time, carbon, and price costs.
- Two workflows are covered: Homework-style queries (Q1–Q5, Q6–Q7) and a final selection challenge.
- Database signatures DB1–DB5 encode different denormalization choices around Product, Stock, and OrderLine.
- The cost model is IO-driven, with an explicit network multiplier for wide scans and shuffles.
- Sharding awareness reduces scan or shuffle costs when filters or group-by align with the shard key.
- For Q1–Q5, all DB signatures use the same collection sizes in the simulator configuration.
- As a result, Q1–Q5 totals are identical across DB1–DB5 under current parameters.
- Aggregation results show a large benefit when grouping aligns with the shard key.
- The final challenge compares DB1 vs DB2 on a Movie/Review workload.
- DB1 wins across time, carbon, and price due to a smaller Review collection.
- The recommended overall data model is DB1 for the provided workloads and assumptions.

## Context and Objectives
Data model selection matters because query performance, energy use, and cost depend heavily on
document sizes, join paths, and how often data must be reshuffled across a cluster. The project
optimizes three metrics simultaneously: execution time, carbon footprint, and price. The simulator
translates schema designs and workload assumptions into IO scans, network shuffles, and aggregated
costs so the trade-offs between denormalizations can be compared in a consistent way.

## Project Scope and Assumptions
- Cluster assumptions:
  - 1000 servers and sharding access fraction 0.1 for targeted lookups.
  - Network overhead multiplier of 5.0 for multi-node scans/shuffles.
- Workload assumptions:
  - Q1–Q5 filters/joins use fixed selectivities from configuration.
  - Challenge queries include frequencies from `data/chapter5/queries.json`.
- Data size/statistics assumptions:
  - Base entity counts and selectivities are in `query_simulator/config.py`.
  - Challenge stats in `data/chapter5/stats.json` provide document counts and cardinalities.
  - Document sizes are derived from JSON schemas plus array-length statistics.

## Data Models and Denormalization Strategies
Database signatures for the homework workloads are defined in `schemas/db1.json`–`schemas/db5.json`.
Each signature uses JSON Schema references and embedded objects to represent denormalization.

- DB1 (baseline, normalized):
  - Collections: Product, Stock, Warehouse, OrderLine, Client.
  - References: Stock stores `IDP`; OrderLine stores `IDP`; Product is separate.
  - Sharding keys: not explicit in schema; modeled via targeted shard flags in queries.
  - Pros: smaller documents, cheaper scans. Cons: joins needed for product details.
- DB2 (embed Stock in Product):
  - Collections: Product (with `stocks[]`), Warehouse, OrderLine, Client.
  - Stock data embedded under each product document.
  - Pros: product+stock lookups without joins. Cons: larger Product docs, duplication.
- DB3 (embed Product in Stock):
  - Collections: Stock (with embedded Product), Warehouse, OrderLine, Client.
  - Pros: stock queries carry product details. Cons: Stock documents become large.
- DB4 (embed Product in OrderLine):
  - Collections: OrderLine (with embedded Product), Stock, Warehouse, Client.
  - Pros: order queries carry product details. Cons: OrderLine becomes heavy.
- DB5 (embed OrderLine in Product):
  - Collections: Product (with `orderLines[]`), Stock, Warehouse, Client.
  - Pros: product-centric analytics. Cons: Product can balloon with order history.

For the challenge dataset:
- DB1: separate Movie and Review collections.
- DB2: Review embeds `movieTitle` and `movieGenre` for read-time joins.

## Operators Implemented
- Filter operator:
  - Input: collection or intermediate step; optional filter key and selectivity.
  - Sharding-aware: if filter key equals sharding key, scan fraction is multiplied by 0.1.
  - Output: reduced document count and output size.
- Join operator:
  - Two inputs (collections or step outputs) with a join key.
  - Cost model scans both sides; shuffle is avoided if both inputs share the join shard key.
  - Output size is computed from selected output fields.
- Aggregate operator:
  - Map/Shuffle/Reduce style with optional sharding alignment.
  - Sharding-aware: grouping on shard key avoids shuffle.
  - Output document count derived from grouping cardinalities.

Key parameters:
- `BASE_IO_TIME_UNIT`, `BASE_IO_CARBON_UNIT`, `BASE_IO_PRICE_UNIT` in `query_simulator/config.py`.
- `NETWORK_MULTIPLIER` for shuffles and full-cluster scans.
- `SHARDING_ACCESS_FRACTION` for targeted shard access.

## Query Planning and Complex Queries
Queries are translated into operator plans. The challenge workflow uses
`chapter5/planner.py` to map high-level queries into explicit step sequences.

Example plan A (Q5_top_movies_with_titles):
1. Aggregate `Review` by `movieId` to compute `avg_rating`.
2. Join the aggregate with `Movie` on `movieId` to fetch titles.

Example plan B (Q3_movie_review_join):
1. Join `Movie` and `Review` on `movieId` with sharding alignment.

For homework filters and joins (Q1–Q5 in `query_simulator/queries.py`),
each query is defined as a list of collection components with selectivities,
then estimated as either filter or join.

## Cost Model
The cost model is IO-driven:
- Data scanned in GB is multiplied by base units for time, carbon, and price.
- Network overhead multiplies costs by `NETWORK_MULTIPLIER` during shuffles.
- Targeted shard access multiplies scan fractions by `SHARDING_ACCESS_FRACTION`.

Main parameters and sources:
- Collection sizes: derived from `DOC_SIZES_BYTES` and entity counts (homework Q1–Q5).
- Schema-driven sizes: `compute_document_size_bytes` + stats for challenge queries.
- Selectivities and cardinalities: `STATS` or per-collection stats in `stats.json`.

## Experiments
### Homework Q1–Q5 (filters + joins)
- Queries: Q1–Q5 from `query_simulator/queries.py`.
- Denormalizations: DB1–DB5 (schemas in `schemas/`).
- Commands:
  - `python main.py --db DB1`
  - `python main.py --db DB1 --json`

### Homework Q6–Q7 (aggregations)
- Q6: group Stock by `IDP` (aligned with sharding).
- Q7: group OrderLine by `date` (not aligned with sharding).
- Code path: `query_simulator/aggregate.py` with the same stats used by tests.

### Final challenge (Chapter 5)
- Dataset: Movie/Review/User schemas in `data/chapter5/schemas/`.
- Queries: five plans from `data/chapter5/queries.json`.
- Denormalizations compared: DB1 vs DB2.
- Command: `python -m chapter5 --dbs DB1 DB2`.

## Results
### Homework Q1–Q5 totals per DB signature
The current simulator configuration uses the same collection sizes for DB1–DB5,
so totals are identical across DB signatures.

| DB | Total Time | Total Carbon | Total Price |
| --- | --- | --- | --- |
| DB1 | 76.5497 | 38.2748 | 7.6550 |
| DB2 | 76.5497 | 38.2748 | 7.6550 |
| DB3 | 76.5497 | 38.2748 | 7.6550 |
| DB4 | 76.5497 | 38.2748 | 7.6550 |
| DB5 | 76.5497 | 38.2748 | 7.6550 |

Per-query breakdown (DB1; identical for DB2–DB5):

| Query | Time | Carbon | Price |
| --- | --- | --- | --- |
| Q1 | 0.0000 | 0.0000 | 0.0000 |
| Q2 | 0.0002 | 0.0001 | 0.0000 |
| Q3 | 25.5157 | 12.7578 | 2.5516 |
| Q4 | 0.0020 | 0.0010 | 0.0002 |
| Q5 | 51.0317 | 25.5159 | 5.1032 |

### Homework Q6–Q7 aggregation costs
| Query | Variant | Time | Carbon | Price | Notes |
| --- | --- | --- | --- | --- | --- |
| Q6 (Stock by product) | With sharding | 2.8349 | 1.4175 | 0.2835 | Shuffle avoided |
| Q6 (Stock by product) | Without sharding | 25.1867 | 12.5933 | 2.5187 | Shuffle cost dominates |
| Q7 (Orders by date) | Without sharding | 879.2746 | 439.6373 | 87.9275 | Shuffle present |

### Final challenge totals (weighted by query frequencies)
| DB | Total Time | Total Carbon | Total Price |
| --- | --- | --- | --- |
| DB1 | 4.2549 | 2.1275 | 0.4255 |
| DB2 | 11.1095 | 5.5547 | 1.1109 |

Per-query (time/carbon/price):

| Query | DB1 | DB2 |
| --- | --- | --- |
| Q1_movies_by_genre | 0.0011 / 0.0005 / 0.0001 | 0.0011 / 0.0005 / 0.0001 |
| Q2_reviews_for_movie | 0.0000 / 0.0000 / 0.0000 | 0.0000 / 0.0000 / 0.0000 |
| Q3_movie_review_join | 1.0651 / 0.5325 / 0.1065 | 2.7787 / 1.3893 / 0.2779 |
| Q4_avg_rating_by_movie | 1.0468 / 0.5234 / 0.1047 | 2.7604 / 1.3802 / 0.2760 |
| Q5_top_movies_with_titles | 1.0725 / 0.5363 / 0.1073 | 2.7861 / 1.3931 / 0.2786 |

Best DB per metric: DB1 for time, carbon, and price. Recommended overall: DB1.

## Discussion
DB1 consistently outperforms DB2 in the final challenge because DB2 increases
Review document sizes by embedding movie metadata, inflating scans for review-heavy
queries (Q3–Q5). For the homework workload, the simulator uses equal collection sizes
for DB1–DB5, masking the expected denormalization trade-offs. When those sizes are
customized per schema, DB2–DB5 would shift cost between joins and scans.

Sensitivity considerations:
- Higher frequency of join-heavy queries would favor denormalizations that reduce joins,
  even if document sizes grow.
- Larger OrderLine or Stock sizes will amplify Q3/Q5 costs and increase the advantage of
  sharding-aligned filters and aggregates.

Simulator limitations:
- Joins are modeled as full scans plus optional shuffles (no indexing or hash join cost).
- Selectivities are static constants, not data-driven.
- Network costs are proportional to IO volumes with a fixed multiplier.

## Conclusion and Future Work
Under the current assumptions, DB1 is the recommended data model because it minimizes
scan volume while keeping shuffling limited. Future improvements include:
- Implementing alternative join algorithms (hash join, indexed lookup).
- Adding index-aware selectivity and latency models.
- Making collection sizes per DB signature explicit in the homework simulator.
- Modeling caching effects and storage layout (row vs column, compression).

## Appendix
### How to run
```bash
# Homework Q1–Q5
python main.py --db DB1
python -m query_simulator --db DB1 --json

# Final challenge (Chapter 5)
python -m chapter5 --dbs DB1 DB2
```

### Configuration files
- `query_simulator/config.py`: global stats, document sizes, and cost constants.
- `query_simulator/queries.py`: Q1–Q5 definitions and selectivities.
- `schemas/db1.json`–`schemas/db5.json`: homework denormalizations.
- `data/chapter5/schemas/DB1.json`, `data/chapter5/schemas/DB2.json`: challenge schemas.
- `data/chapter5/stats.json`: cluster and collection statistics for challenge.
- `data/chapter5/queries.json`: challenge query list and frequencies.

### Extra details for grading
The aggregate operator uses a Map/Shuffle/Reduce cost model. If grouping aligns
with the sharding key, shuffle is avoided; otherwise, shuffle volume is proportional
to group cardinality and partial aggregate size.
