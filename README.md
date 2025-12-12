# Big Data Infrastructure and Cloud
## Homework 2 — QuerySimulator (Q1–Q5)

`QuerySimulator` automates the cost calculation (time, carbon, price) for the
Filter and Join queries Q1–Q5 from Homework 2 section 3.3. The package reuses
the STATS and sizing constants from Homework 1 so it can run standalone.

---

## What it does
- Computes IO-driven cost for each query component and adds network overheads for joins.
- Uses `STATS` (entity counts, selectivities) and `DOC_SIZES_BYTES` (bytes per base document).
- Supports DB1–DB5 via `collection_size_gb(collection, db)`.
- Emits either a concise text report or JSON for downstream analysis.

---

## Package layout
- `query_simulator/config.py` — STATS, DOC_SIZES_BYTES, cost constants, sizing helpers.
- `query_simulator/models.py` — dataclasses for query specs and cost breakdowns.
- `query_simulator/costs.py` — cost model implementation (IO + network multipliers).
- `query_simulator/queries.py` — default Q1–Q5 definitions with selectivities.
- `query_simulator/runner.py` — CLI runner, text/JSON formatting helpers.
- `query_simulator/__main__.py` — `python -m query_simulator` entrypoint.
- `main.py` — delegates to the QuerySimulator CLI for convenience.
- `schemas/*.json` — JSON schemas from Homework 1 (unchanged).

---

## Key constants (config.py)
- `STATS`: entity counts and selectivities (`SEL_Q1_STOCK`, `SEL_Q2_BRAND`, `SEL_Q3_DATE`).
- `DOC_SIZES_BYTES`: un-denormalized document sizes in bytes.
- Cost parameters: `BASE_IO_TIME_UNIT`, `BASE_IO_CARBON_UNIT`, `BASE_IO_PRICE_UNIT`,
  `NETWORK_MULTIPLIER`, `SHARDING_ACCESS_FRACTION`.
- `collection_size_gb(collection, db)` pulls sizes for DB1–DB5 (same baseline by default).

---

## Queries covered
- **Q1_stock_lookup**: stock for one product in one warehouse (targeted shard access).
- **Q2_brand_filter**: Apple products by brand.
- **Q3_orders_by_date**: order lines for a specific date.
- **Q4_brand_stock_join**: Apple products joined with their stock entries.
- **Q5_orders_brand_client_join**: orders on a date joined with Apple products and client data.

---

## How to run
1) (Optional) Activate a virtual environment (on Windows Powershell).
```
python -m venv venv
.\venv\Scripts\Activate.ps1
```
2) Install requirements: `pip install -r requirements.txt`
3) Run the simulator (text):
```bash
python main.py --db DB1
# or
python -m query_simulator --db DB1
```
4) JSON output for further processing:
```bash
python -m query_simulator --db DB1 --json
```

---

## Notes
- All sizes are derived from Homework 1 placeholders; adjust `DOC_SIZES_BYTES` or STATS as needed.
- The network multiplier and shard access fraction are surfaced to make trade-offs clear.
