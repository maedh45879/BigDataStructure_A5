# Big Data Infrastructure and Cloud  
## 🧮 Homework 2.7 – NoSQL Data Model Simulation  

This project simulates the **storage size** and **data distribution (sharding)** of different NoSQL database models.  
It is part of the *Big Data Infrastructure and Cloud* course.  

---

## 🎯 Project Description  

The goal is to create a Python program that:  
1. Reads a **JSON Schema** (structure of a collection).  
2. Uses given **statistics** (number of documents, array lengths, etc.).  
3. Computes:  
   - The size of a document (in bytes)  
   - The size of a collection (in GB)  
   - The total database size  
4. Simulates **sharding** over a cluster of servers to compute:  
   - Average number of documents per server  
   - Average number of distinct key values per server  

---

## 👥 Team Members  

| Name | Role | Main Tasks |
|------|------|------------|
| **Manon AUBRY** | JSON Schemas | Create DB1–DB5 JSON schemas and validate them |
| **Devaraj RAMAMMOORTHY** | Size Computation | Develop functions for document, collection, and database size |
| **Sandeep PIDUGU** | Sharding & Integration | Implement sharding simulation and integrate all modules |

---

## 🗂️ Project Structure  

```bash
project/
│
├── main.py                  # Main Python program
├── compute_sizes.py         # Size calculation module
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

## 🚀 How to Run the Project

### 1️⃣ Prerequisites

Make sure you have **Python 3.10+** installed.

### 2️⃣ Create and Activate a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # On Windows
# source .venv/bin/activate   # On macOS/Linux
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Project

```bash
python main.py
```

You should see output similar to:

```
Document size (bytes): 1056
Collection size (GB): 0.09834766387939453
Database total size (GB): 0.09834766387939453
{'collection': 'Prod', 'sharding_key': 'IDP', 'nb_servers': 1000, 'avg_docs_per_server': 100.0, 'avg_distinct_key_values_per_server': 100.0}
{'collection': 'Prod', 'sharding_key': 'brand', 'nb_servers': 1000, 'avg_docs_per_server': 100.0, 'avg_distinct_key_values_per_server': 5.0}
```

### 5️⃣ Exit the Virtual Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```
