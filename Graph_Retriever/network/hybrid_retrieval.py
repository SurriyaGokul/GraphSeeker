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

# Essentially the same but doesnt take label_diff as not needed during inference
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
    cross_edges = torch.tensor(cross_edges, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, cross_edges], dim=1)

    cross_attr = torch.zeros((cross_edges.size(1), edge_attr.size(1)))
    edge_attr = torch.cat([edge_attr, cross_attr], dim=0)

    batch = torch.zeros(N_q + N_c, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

class HybridRetrievalSystem:
    def __init__(self, graphs, labels, embeddings):
        self.graphs = graphs
        self.labels = labels
        self.embeddings = embeddings
        self.embedding_dim = embeddings.shape[1]

        self.re_ranker = CrossEncoderGNN(node_dim=64, edge_dim=5).to(device)
        self.re_ranker.load_state_dict(torch.load(RE_RANKER_WEIGHTS, map_location=device))
        self.re_ranker.eval()

        self.faiss_index = self.build_index(embeddings)

    def build_index(self, db_embeddings):
        db_embeddings = np.ascontiguousarray(db_embeddings, dtype=np.float32)
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, NLIST, M, NBITS)

        print("Training FAISS index")
        index.train(db_embeddings)
        index.add(db_embeddings)
        index.nprobe = NPROBE
        print(f"FAISS index trained and built with {index.ntotal} items.")
        return index

    def search_and_rerank(self, num_queries=100):

        for i in tqdm(range(num_queries), desc="🔍 Querying and Re-ranking"):
            q_emb = self.embeddings[i].reshape(1, -1)
            q_label = self.labels[i]
            q_graph = self.graphs[i]

            _, top_k_indices = self.faiss_index.search(q_emb, TOP_K_CANDIDATES)
            top_k_indices = top_k_indices.flatten()

            rerank_scores = []
            for idx in top_k_indices:
                if idx == i:
                    continue  # Skip itself
                c_graph = self.graphs[idx]
                joint_graph = build_joint_graph(q_graph, c_graph).to(device)

                with torch.no_grad():
                    pred_diff = self.re_ranker(joint_graph.x, joint_graph.edge_index, joint_graph.edge_attr, joint_graph.batch)
                    pred_diff = abs(pred_diff.item())  # Closer label = more similar
                    rerank_scores.append((idx, pred_diff))

            rerank_scores.sort(key=lambda x: x[1])
            top_indices = [idx for idx, _ in rerank_scores[:TOP_N_RESULTS]]

            return top_indices, top_k_indices

if __name__ == "__main__":

    # Load the dataset and the pretrained embeddings
    dataset = ZINC(root="data/ZINC", subset=False, split="train")
    graphs = list(dataset)

    with open("/teamspace/studios/this_studio/embeddings/train_graph_embeddings.pkl", "rb") as f:
        emb_data = pickle.load(f)
    embeddings = np.stack([v["embedding"] for v in emb_data.values()]).astype("float32")
    labels = np.array([v["label"] for v in emb_data.values()])

    assert len(graphs) == len(labels) == len(embeddings) # simple sanity check
    system = HybridRetrievalSystem(graphs, labels, embeddings)
    top_indices, top_k_indices = system.search_and_rerank(num_queries=NUM_QUERY)
    print(f"Top indices for query 0: {top_indices}")
    print(f"Top K indices from FAISS: {top_k_indices}")
