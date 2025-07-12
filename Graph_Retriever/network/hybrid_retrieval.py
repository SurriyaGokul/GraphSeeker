import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
import time
import yaml
import pickle
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.datasets import ZINC
from re_ranker import CrossEncoderGNN
from utils.atom_encoder import SimpleAtomEncoder
from utils.graph_utils import preprocess_graph
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration
with open("/teamspace/studios/this_studio/Graph_Retriever/config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

NUM_QUERY = config["NUM_QUERY"]
EMBEDDING_DIM = config["EMBEDDING_DIM"]
NLIST = config["NLIST"]
M = config["M"]
NBITS = config["NBITS"]
NPROBE = config["NPROBE"]
TOP_K_CANDIDATES = config["TOP_K_CANDIDATES"]
TOP_N_RESULTS = config["TOP_N_RESULTS"]
RE_RANKER_WEIGHTS = config["RE_RANKER_WEIGHTS"]

atom_encoder = SimpleAtomEncoder(emb_dim = 64).to(device)

# Function to build joint graph for query and candidate
def build_joint_graph(q_data, c_data):
    q_data, c_data = preprocess_graph(q_data), preprocess_graph(c_data)
    N_q, N_c = q_data.num_nodes, c_data.num_nodes

    x = torch.cat([q_data.x, c_data.x], dim=0)
    edge_index = torch.cat([q_data.edge_index, c_data.edge_index + N_q], dim=1)
    edge_attr = torch.cat([q_data.edge_attr, c_data.edge_attr], dim=0)

    cross_edges = []
    for i in range(N_q):
        for j in range(N_c):
            cross_edges.append([i, N_q + j])
            cross_edges.append([N_q + j, i])
    
    cross_edges = torch.tensor(cross_edges, dtype=torch.long, device=device).t()
    cross_attr = torch.zeros((cross_edges.size(1), edge_attr.size(1)), device=device)

    edge_index = torch.cat([edge_index, cross_edges], dim=1)
    edge_attr = torch.cat([edge_attr, cross_attr], dim=0)

    batch = torch.zeros(N_q + N_c, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

# Hybrid Retrieval System
class HybridRetrievalSystem:
    def __init__(self, graphs, labels, embeddings):
        self.graphs = graphs
        self.min_label = np.min(labels)
        self.max_label = np.max(labels)
        self.labels = 2 * (labels - self.min_label) / (self.max_label - self.min_label) - 1
        self.embeddings = embeddings
        self.embedding_dim = embeddings.shape[1]

        self.re_ranker = CrossEncoderGNN(node_dim=128, edge_dim=5).to(device)
        self.re_ranker.load_state_dict(torch.load(RE_RANKER_WEIGHTS, map_location=device))
        self.re_ranker.eval()

        self.faiss_index = self.build_index(embeddings)

    def build_index(self, db_embeddings):
        db_embeddings = np.ascontiguousarray(db_embeddings, dtype=np.float32)
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, NLIST, M, NBITS)
        index.train(db_embeddings)
        index.add(db_embeddings)
        index.nprobe = NPROBE
        print(f"FAISS index trained and built with {index.ntotal} items.")
        return index

    def search_and_rerank(self, num_queries, query_graphs, query_labels, query_embeddings):
        recall = 0
        mrr = 0
        ranks = []
        total_time = 0
        query_labels = 2 * (query_labels - self.min_label) / (self.max_label - self.min_label) - 1

        for i in tqdm(range(num_queries), desc="Querying and Re-ranking"):
            start_time = time.time()

            q_emb = query_embeddings[i].reshape(1, -1)
            q_label = query_labels[i]
            q_graph = query_graphs[i]

            _, top_k_indices = self.faiss_index.search(q_emb, TOP_K_CANDIDATES)
            top_k_indices = top_k_indices.flatten()

            rerank_scores = []
            for idx in top_k_indices:
                c_graph = self.graphs[idx]
                joint_graph = build_joint_graph(q_graph, c_graph).to(device)

                with torch.no_grad():
                    pred_diff = self.re_ranker(joint_graph.x, joint_graph.edge_index, joint_graph.edge_attr, joint_graph.batch)
                    pred_diff = abs(pred_diff.item())  # Smaller predicted diff → more similar
                    rerank_scores.append((idx, pred_diff))

            rerank_scores.sort(key=lambda x: x[1])
            top_indices = [idx for idx, _ in rerank_scores[:TOP_N_RESULTS]]

            
            correct = [
                abs(q_label - self.labels[j]) < 0.03
                for j in top_indices
            ]
            if any(correct):
                recall += 1

            for rank, j in enumerate(top_indices):
                if abs(q_label - self.labels[j]) < 0.03:
                    mrr += 1 / (rank + 1)
                    ranks.append(rank + 1)
                    break

            total_time += (time.time() - start_time)

        recall_score = recall / num_queries
        mrr_score = mrr / num_queries
        avg_time_per_query = total_time / num_queries

        print(f"\n Final Evaluation (Top-{TOP_N_RESULTS}):")
        print(f"Recall@{TOP_N_RESULTS}: {recall_score:.4f}")
        print(f"MRR: {mrr_score:.4f}")
        if ranks:
            print(f"Mean Rank of first correct result: {np.mean(ranks):.2f}")
        print(f"Average Time per Query: {avg_time_per_query:.4f} seconds")



# ---------- Main ----------
if __name__ == "__main__":

    # Load dataset and embeddings
    dataset = ZINC(root="data/ZINC", subset=False, split="train")
    graphs = list(dataset)

    with open("/teamspace/studios/this_studio/embeddings/train_graph_embeddings.pkl", "rb") as f:
        emb_data = pickle.load(f)
    embeddings = np.stack([v["embedding"] for v in emb_data.values()]).astype("float32")
    labels = np.array([v["label"] for v in emb_data.values()])

    assert len(graphs) == len(labels) == len(embeddings)

    # Split into query and train sets (90% index, 10% query)
    indices = np.arange(len(graphs))
    train_idx, query_idx = train_test_split(indices, test_size=0.1, random_state=42)

    
    index_graphs = [graphs[i] for i in train_idx]
    index_labels = labels[train_idx]
    index_embeddings = embeddings[train_idx]

    query_subset = query_idx[:NUM_QUERY]
    query_graphs = [graphs[i] for i in query_subset]
    query_labels = labels[query_subset]
    query_embeddings = embeddings[query_subset]


    # Run retrieval pipeline
    system = HybridRetrievalSystem(index_graphs, index_labels, index_embeddings)
    system.search_and_rerank(
        num_queries=len(query_graphs),
        query_graphs=query_graphs,
        query_labels=query_labels,
        query_embeddings=query_embeddings,
    )

