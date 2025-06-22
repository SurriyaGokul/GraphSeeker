#  Siamese Graph Transformer for Large-Scale Graph Retrieval

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning system for **graph similarity search at scale**, built entirely from scratch. This project combines a **custom transformer-based Graphormer**, a **FAISS IVF+PQ index**, and a **learned re-ranking model** to retrieve highly similar graphs from massive datasets like **OGB-MolPCBA** in milliseconds.

---

##  Highlights

-  **Fully Custom Graph Transformer**: Implements a transformer architecture from scratch for graph-structured data, including positional encodings, attention masking, and learned graph embeddings.
-  **Siamese Contrastive Training**: Trains a Siamese model to learn a discriminative embedding space where similar graphs are close and dissimilar graphs are far apart.
-  **High-Speed ANN Search**: Uses FAISS’s **IVF+PQ** index to achieve scalable and efficient retrieval across 350K+ molecular graphs.
-  **Learned Re-ranking**: Applies a lightweight MLP trained on Jaccard similarity over molecular labels to rerank candidates and boost retrieval precision.
-  **Tested on OGB-MolPCBA**: A multi-label classification benchmark used to simulate real-world graph retrieval tasks.

---

##  Architecture Overview

### 1. Graph Embedding (Training Phase)

```text
(Graph A, Graph B) ──────> [Shared Graphormer Encoder] ─────> Embedding A, Embedding B
                                                └─────> Contrastive Loss
````

* Graph pairs are passed through a **shared-weight transformer encoder**.
* A **contrastive loss** is used to pull together similar graph embeddings and push apart dissimilar ones.

### 2. Embedding & Indexing (Post-training)

```text
Graph Dataset ──> [Trained Graphormer] ──> Embeddings ──> [FAISS IVF+PQ Index]
```

* All graph embeddings are computed and indexed using **Product Quantization** in FAISS for sublinear search time.

### 3. Retrieval + Re-ranking

```text
Query Graph ──> [Trained Graphormer] ──> Query Embedding ──┐
                                                          │
                                               [FAISS Search]
                                                          │
                                Top-K Candidates ─────────┘
                                                          ↓
                          [Neural Re-Ranker (MLP)] ──> Final Sorted Results
```

---

## 🚀 Getting Started

### 📦 Requirements

* Python ≥ 3.8
* PyTorch ≥ 1.10
* FAISS (GPU strongly recommended)
* NumPy, PyYAML, OGB

```bash
pip install -r requirements.txt
```

### ⚙️ Dataset

We use the **OGB-MolPCBA** dataset from the [Open Graph Benchmark](https://ogb.stanford.edu/docs/graphprop/). It contains 437k+ molecular graphs with multi-label annotations.

---

##  Usage

### 1. **Train the Graphormer Encoder**

Train your Siamese model using contrastive loss on graph pairs.

```bash
python train.py
```

### 2. **Generate Embeddings**

After training, generate and save embeddings for the full dataset.

```bash
python generate_embeddings.py
```

### 3. **Build FAISS Index**

Build an IVF+PQ FAISS index using the saved embeddings.

```bash
python build_index.py
```

### 4. **Train the Re-ranker**

Use Jaccard similarity over MolPCBA labels to train the reranking MLP.

```bash
python train_reranker.py
```

### 5. **Run Retrieval**

Run a query graph through the trained encoder and retrieve top-K similar graphs.

```bash
python retrieve.py --query_id 42
```

---

## 📊 Evaluation

We use label overlap (e.g., Jaccard similarity) between query and retrieved graphs as a proxy for retrieval relevance.

* Precision\@K
* Recall\@K
* Average Jaccard of Top-K

(Coming soon: scripts for automated retrieval evaluation.)

---

## Example Output

```text
--- Query Graph: #42 ---
Top 10 Retrieved:
Graph ID: 21812 | Score: 0.9312
Graph ID: 11156 | Score: 0.9078
Graph ID: 76820 | Score: 0.8765
...
```

---

## Acknowledgements

* [OGB Benchmark](https://ogb.stanford.edu/)
* [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)
* General inspiration from transformer-based architectures and representation learning research.

---

## License

MIT License © \[Your Name]

---

## Contributing

Pull requests are welcome! Feel free to open issues or contribute new features (e.g., better re-ranking models, GAT support, batch retrieval).
