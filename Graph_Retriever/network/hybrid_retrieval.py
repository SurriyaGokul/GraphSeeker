import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import faiss
import time
import yaml
from re_ranker import CrossEncoderGNN

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

    def search(self, query_embedding: np.ndarray):
        if self.faiss_index is None:
            raise RuntimeError("Index not built. Call `build_index` first.")

        if not query_embedding.flags['C_CONTIGUOUS']:
            query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

        self.faiss_index.nprobe = NPROBE

        start_time = time.time()
        distances, indices = self.faiss_index.search(query_embedding, TOP_K_CANDIDATES)
        stage1_time = time.time() - start_time

        candidate_indices = indices[0]
        query_tensor = torch.from_numpy(query_embedding).float()
        candidate_embeddings = self.db_embeddings[candidate_indices]
        query_batch = query_tensor.repeat(len(candidate_embeddings), 1)
        candidate_embeddings = candidate_embeddings.float()
        return candidate_indices, distances[0]

if __name__ == '__main__':
    print("Loading labels and graph embeddings...")
    query_embeddings = np.random.random((NUM_QUERY, EMBEDDING_DIM)).astype('float32')
    faiss.normalize_L2(query_embeddings)

