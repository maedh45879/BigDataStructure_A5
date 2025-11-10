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
| **Manon Aubry** | JSON Schemas | Create DB1–DB5 JSON schemas and validate them |
| **Devaraj RAMAMMOORTHY** | Size Computation | Develop functions for document, collection, and database size |
| **Sandeep PIDUGU** | Sharding & Integration | Implement sharding simulation and integrate all modules |

---

## 🗂️ Project Structure  

```bash
project/
│
├── main.py          # Main Python program
├── compute_sizes.py          # Size calculation module
├── schemas/                  # Folder containing all JSON schemas
│   ├── db1.json
│   ├── db2.json
│   ├── db3.json
│   ├── db4.json
│   ├── db5.json
│   └── product.json
└── README.md
