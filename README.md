# **GraphSeeker**

### *Scalable Graph Retrieval with Siamese Graph Transformers and GNN-Based Reranking*
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Built with PyTorch](https://img.shields.io/badge/Built%20with-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-Framework-brightgreen?logo=github)](https://pytorch-geometric.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Dataset-OGB--MolHIV-purple)](https://ogb.stanford.edu/docs/graphprop/)
[![Project Status](https://img.shields.io/badge/Status-Research--grade-success)]()

---
**GraphSeeker** is a high-performance graph retrieval system engineered for scalability and precision. It integrates a **Siamese Graph Transformer encoder** with a **two-stage hybrid retrieval pipeline**, combining FAISS-based dense retrieval with GNN-based reranking. The result: accurate identification of **structurally and semantically similar graphs** at scale.

---

##  Key Features

###  **Siamese Graph Transformer Encoder**

A dual-branch transformer architecture trained via **contrastive learning** to produce rich, discriminative **graph-level embeddings**.

* **Edge-Conditioned Attention**
  Incorporates edge attributes into the attention mechanism to capture **node–edge–subgraph** interactions.

* **Virtual Node for Global Context**
  Adds a learnable **virtual token** to each graph to represent global structure in a **size-invariant** manner.

* **Pairwise Contrastive Training**
  Trained using **contrastive loss**, robust to class imbalance—ideal for tasks like **retrieval**, **matching**, and **clustering**.

* **Task-Agnostic Embeddings**
  Embeddings can be transferred to a wide range of downstream tasks with **minimal fine-tuning**, including **few-shot classification** and **unsupervised clustering**.

---

###  **Graph Retrieval Pipeline**

####  FAISS-Based Approximate Nearest Neighbor Search (IVF + PQ)

* Scales to **tens of thousands** of graph embeddings with sublinear retrieval time
* Utilizes `IVF` (Inverted File Indexing) and `PQ` (Product Quantization) for fast, memory-efficient similarity search
* Tunable parameters (`NLIST`, `NBITS`, `NPROBE`) allow for customizable **speed/recall trade-offs**
* Supports optional **L2 normalization** for cosine similarity

---

####  Cross-Encoder GNN Reranker

* **Reranks top-K FAISS candidates** using a GNN-based cross-encoder
* Merges query and candidate into a **joint graph** with **cross-edges**
* Applies **message passing** and **graph-level pooling** for final scoring
* Designed with **balanced supervision** to address dataset imbalance

---

####  Dynamic Joint Graph Construction

* Dynamically merges graphs during inference with **inter-graph edges**
* Captures **fine-grained structural similarities** across graphs
* Enables reranker to exploit **both local substructures and global context**

---

##  Architecture Overview

```
Query Graph
    |
    v
[Siamese Graph Transformer Encoder]
    |
    v
Graph-Level Embedding
    |
    v
[FAISS IVF+PQ ANN Retrieval]
    |
    v
Top-K Candidate Graphs
    |
    v
[Cross-Encoder GNN Reranker]
    |
    v
Final Ranked Results
```

---

## 📂 Project Structure

```
Scalable_Graph_Retrieval/
  ├── Siamese-Graphormer/
  │   ├── generate_embeddings.py
  │   ├── loss.py
  │   ├── model.py
  │   ├── train.py
  │   ├── data/
  │   │   └── dataset.py
  │   ├── loss/
  │   │   └── loss.py
  │   └── network/
  │       ├── edge_attention.py
  │       ├── encoder.py
  │       └── siamese.py
  ├── Graph_Retriever/
  │   ├── get_similiar.py
  │   ├── train_re_ranker.py
  │   ├── config/
  │   │   └── config.yaml
  │   ├── embeddings/
  │   │   ├── check.ipynb
  │   │   └── graph_embeddings.pt
  │   └── network/
  │       ├── hybrid_retrieval.py
  │       └── re_ranker.py
  └── README.md
```

---

## ⚡ Quickstart

```bash
# 1. Train the Siamese Graph Transformer and generate graph embeddings
python train.py

# 2. Build the FAISS index and perform hybrid retrieval with GNN-based reranking
python Graph_Retriever/get_similiar.py
```

---

## 📊 Dataset

* Utilizes the **OGB-MolHIV** benchmark for supervised training
* Easily adaptable to any graph classification dataset with **graph-level labels**

---

## 🤝 Contributing

We welcome contributions!
Feel free to open issues or submit PRs to extend **GraphSeeker** to new graph domains, benchmarks, or retrieval strategies.

---

## 📄 License

Distributed under the **MIT License**.

---

