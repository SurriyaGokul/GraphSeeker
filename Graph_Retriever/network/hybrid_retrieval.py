import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import faiss
import time
import yaml
from re_ranker import CrossEncoderGNN
from torch_geometric.data import Data
from train_re_ranker import load_graph_dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load configuration
with open("/content/Graph_Retriever/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

NUM_GRAPHS = config['NUM_GRAPHS']
EMBEDDING_DIM = config['EMBEDDING_DIM']
NUM_QUERY = config['NUM_QUERY']
NLIST = config['NLIST']
M = config['M']
NBITS = config['NBITS']
NPROBE = config['NPROBE']
TOP_K_CANDIDATES = config['TOP_K_CANDIDATES']
TOP_N_RESULTS = config['TOP_N_RESULTS']

class HybridRetrievalSystem:
    def __init__(self, embedding_dim,training=False):
        self.embedding_dim = embedding_dim
        self.faiss_index = None
        self.re_ranker = CrossEncoderGNN(embedding_dim)
        self.re_ranker.eval()
        self.db_embeddings = None
        self.graphs = load_graph_dataset() 
        if not training:
            self.reranker.load_state_dict(torch.load("/content/Graph_Retriever/reranker.pth", map_location='cpu'))

    def build_index(self, db_embeddings: np.ndarray):
        if not db_embeddings.flags['C_CONTIGUOUS']:
            db_embeddings = np.ascontiguousarray(db_embeddings, dtype=np.float32)

        self.db_embeddings = torch.from_numpy(db_embeddings)

        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        self.faiss_index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, NLIST, M, NBITS)

        print(f"Training FAISS index on {len(db_embeddings)} vectors...")
        start_time = time.time()
        self.faiss_index.train(db_embeddings)
        print(f"Training complete in {time.time() - start_time:.2f} seconds.")

        print("Adding embeddings to index...")
        start_time = time.time()
        self.faiss_index.add(db_embeddings)
        print(f"Addition complete in {time.time() - start_time:.2f} seconds.")
        print(f"FAISS index built with {self.faiss_index.ntotal} items.")

    def build_joint_graph(self, q_data, c_data, add_cross_edges=True):
        q_x = q_data.x
        c_x = c_data.x
        N_q = q_x.shape[0]
        N_c = c_x.shape[0]

        joint_x = torch.cat([q_x, c_x], dim=0)
        q_edge_index = q_data.edge_index
        c_edge_index = c_data.edge_index + N_q

        joint_edge_index = torch.cat([q_edge_index, c_edge_index], dim=1)

        if add_cross_edges:
            cross_edges = []
            for q_node in range(N_q):
                for c_node in range(N_c):
                    cross_edges.append([q_node, N_q + c_node])
                    cross_edges.append([N_q + c_node, q_node])
            cross_edges = torch.tensor(cross_edges, dtype=torch.long).t().contiguous()
            joint_edge_index = torch.cat([joint_edge_index, cross_edges], dim=1)

        joint_batch = torch.zeros(N_q + N_c, dtype=torch.long)
        return Data(x=joint_x, edge_index=joint_edge_index, batch=joint_batch)

    def search(self, query_embedding: np.ndarray):
        if self.faiss_index is None:
            raise RuntimeError("Index not built")

        if not query_embedding.flags['C_CONTIGUOUS']:
            query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

        self.faiss_index.nprobe = NPROBE
        distances, indices = self.faiss_index.search(query_embedding, TOP_K_CANDIDATES)

        query_idx = 0
        top_k_indices = indices[query_idx]
        q_embedding = query_embedding[query_idx]
        q_graph = self.graphs[query_idx]

        reranker_scores = []
        for candidate_idx in top_k_indices:
            if candidate_idx >= len(self.graphs):
                continue
            c_graph = self.graphs[candidate_idx]
            joint_graph = self.build_joint_graph(q_graph, c_graph)
            joint_graph = joint_graph.to(device)

            with torch.no_grad():
                score = self.re_ranker(joint_graph.x, joint_graph.edge_index, joint_graph.batch)
                reranker_scores.append((candidate_idx, score.item()))

        # Sort by score descending
        reranker_scores.sort(key=lambda x: x[1], reverse=True)
        top_n = reranker_scores[:TOP_N_RESULTS]
        top_indices, top_scores = zip(*top_n)

        return top_indices, top_scores
        

if __name__ == '__main__':
    print("Loading labels and graph embeddings...")
    query_embeddings = np.random.random((NUM_QUERY, EMBEDDING_DIM)).astype('float32')
    faiss.normalize_L2(query_embeddings)

