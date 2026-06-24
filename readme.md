# 🚀 VectorDB Engine: From-Scratch Indexing & Local RAG Visualizer

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Local AI](https://img.shields.io/badge/Ollama-llama3.2%3A1b-orange.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced, full-stack **Vector Database, Semantic Search, and RAG Engine** built entirely from scratch. This project features an interactive custom UI dashboard that visualizes high-dimensional embeddings mapped into a 2D PCA spatial layout, coupled with an active local LLM pipeline.

---

## 🌟 Core Highlights

* **From-Scratch Multi-Algorithm Indexing:**
  * **HNSW (Hierarchical Navigable Small World):** Custom multi-layer proximity graph construction for fast approximate nearest neighbor retrieval.
  * **KD-Tree:** Dimensional space-partitioning tree built for precise spatial coordinates.
  * **Brute Force:** Pure linear scan mapping exact KNN baselines to dynamically audit benchmark performance.
* **Vector Metrics Support:** Flexible metric evaluation via **Cosine Similarity**, **Euclidean Distance**, and **Manhattan Distance**.
* **Real-time 2D PCA Projection:** High-dimensional vectors (up to 768D from Ollama) are dynamically projected using Principal Component Analysis down to a stunning 2D interactive canvas.
* **Production-Grade Local RAG:** Fully functional Document Chunking, Embedding Creation (`nomic-embed-text`), and Question Answering pipeline powered safely on-device by `llama3.2:1b`.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[User UI Dashboard] -->|1. Raw Query / Document| B(Flask Backend)
    B -->|2. Get Embeddings| C[Local Ollama Instance]
    C -->|3. Return Vectors| B
    B -->|4. Index / Search| D{Vector Storage}
    D -->|HNSW Graph| E[Results & Latency Metrics]
    D -->|KD-Tree / BruteForce| E
    E -->|5. 2D PCA Projection Data| 
    📁 Project Structure
The engine is engineered as a highly lightweight, clean two-file implementation keeping memory foot-print and deployment modularity optimal:

Plaintext
VECTPR DB/
│
├── vector_db_server.py  # Thread-safe Vector Storage Engine, Custom DSA Indexes, & Flask Core REST APIs
└── index.html           # Modern Neon Dashboard, 2D PCA Projection Canvas Engine, & AI I