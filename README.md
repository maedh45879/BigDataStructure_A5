# Big Data Infrastructure and Cloud
## Homework 2.7 – NoSQL Data Model Simulation

This project simulates the storage size and sharding distribution of different NoSQL data models for the Big Data Infrastructure and Cloud course.

---

## Goal

Build a Python tool that, given a JSON schema and a few sizing stats, reports the storage footprint of a collection and simulates sharding to see how evenly data spreads across a cluster.

### Main Computations

1. Size calculation
   - Document size (bytes)
   - Collection size (GB)
   - Total database size
2. Sharding simulation
   - Average number of documents per server
   - Average number of distinct key values per server

### How the code works (by file and function)

- `main.py`
  - `load_json_schema(path)`: opens a JSON schema file from `schemas/` and returns the parsed dict.
  - `CollectionStats`: data class carrying inputs for a collection (document count, average array lengths, sharding key cardinalities).
  - `primitive_value_size(json_type, format_)`: maps primitive JSON types to byte sizes (number/integer = 8B, string = 80B, date strings = 20B, boolean/null = 8B).
  - `compute_document_size_bytes(schema, stats, prefix)`: recursively walks a JSON schema to sum bytes for each field; uses `avg_array_lengths` to expand arrays and adds a small overhead per object property.
  - `compute_collection_size_gb(schema, stats)`: multiplies one document’s bytes by `nb_documents` and converts to GB.
  - `CollectionModel`: wraps a schema and stats; exposes `document_size_bytes()` and `size_gb()` helpers.
  - `DatabaseModel`: holds multiple collections and aggregates `total_size_gb()`.
  - `compute_sharding_stats(coll, sharding_key, nb_servers)`: estimates average documents and distinct sharding-key values per server for a given key.
  - `example()`: loads `schemas/product.json`, fills `CollectionStats`, prints size calculations, and runs sharding simulations for two keys (`IDP`, `brand`).
- `schemas/*.json`
  - JSON schemas defining the structure of each collection (e.g., `product.json`).
- `requirements.txt`
  - Minimal dependencies to run the tool.

---

## Team Contributions

| Name | Role | Main Tasks |
|------|------|------------|
| **Manon AUBRY** | JSON Schemas | Create DB1–DB5 JSON schemas and validate them |
| **Devaraj RAMAMMOORTHY** & **Mehdi MAMLOUK** | Size Computation | Develop functions for document, collection, and database size |
| **Sandeep PIDUGU** | Sharding & Integration | Implement sharding simulation and integrate all modules |

---

## Project Structure

```bash
project/
│
├── main.py                  # Main Python program
├── schemas/                 # Folder containing all JSON schemas
│   ├── db1.json
│   ├── db2.json
│   ├── db3.json
│   ├── db4.json
│   ├── db5.json
│   └── product.json
├── requirements.txt         # Python dependencies
└── README.md
```

---

## How to Run

1. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # On Windows
   # source .venv/bin/activate   # On macOS/Linux
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the simulation
   ```bash
   python main.py
   ```

Expected output (example):
```
Document size (bytes): 1056
Collection size (GB): 0.09834766387939453
Database total size (GB): 0.09834766387939453
{'collection': 'Prod', 'sharding_key': 'IDP', 'nb_servers': 1000, 'avg_docs_per_server': 100.0, 'avg_distinct_key_values_per_server': 100.0}
{'collection': 'Prod', 'sharding_key': 'brand', 'nb_servers': 1000, 'avg_docs_per_server': 100.0, 'avg_distinct_key_values_per_server': 5.0}
```

---

## Conclusion

The tool highlights how schema design and sharding key choice affect storage footprint and distribution across servers—useful insights before deploying at scale.
