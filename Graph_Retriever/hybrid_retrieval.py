import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import faiss
import time
import yaml
from re_ranker import ReRankerNet

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
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.faiss_index = None
        self.re_ranker = ReRankerNet(embedding_dim)
        self.re_ranker.eval()
        self.db_embeddings = None

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

        valid_labels = labels[:len(self.db_embeddings)]
        self._train_reranker(labels=valid_labels)

    def _train_reranker(self, labels: np.ndarray, num_pairs=100000, epochs=30):
        self.re_ranker.train()
        optimizer = optim.Adam(self.re_ranker.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        label_tensor = torch.from_numpy(labels).float()
        available_size = self.db_embeddings.shape[0]

        indices1 = np.random.randint(0, available_size, num_pairs)
        indices2 = np.random.randint(0, available_size, num_pairs)

        q_vectors = self.db_embeddings[indices1]
        c_vectors = self.db_embeddings[indices2]

        y1 = (label_tensor[indices1] > 0)
        y2 = (label_tensor[indices2] > 0)

        intersection = (y1 & y2).sum(dim=1).float()
        union = (y1 | y2).sum(dim=1).float()
        jaccard = (intersection / (union + 1e-8)).unsqueeze(1)
        labels_binary = (jaccard > 0.5).float()

        for epoch in range(epochs):
            optimizer.zero_grad()
            scores = self.re_ranker(q_vectors, c_vectors)
            loss = criterion(scores, labels_binary)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}")

        self.re_ranker.eval()
        print("Re-ranker training complete.")

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

        with torch.no_grad():
            scores = self.re_ranker(query_batch, candidate_embeddings).squeeze()

        sorted_reranked_indices = torch.argsort(scores, descending=True)
        final_indices = candidate_indices[sorted_reranked_indices.numpy()]
        final_scores = scores[sorted_reranked_indices].numpy()

        stage2_time = time.time() - start_time - stage1_time
        print(f"Search complete. Stage 1: {stage1_time*1000:.2f} ms, Stage 2: {stage2_time*1000:.2f} ms")
        return final_indices[:TOP_N_RESULTS], final_scores[:TOP_N_RESULTS]

if __name__ == '__main__':
    print("Loading labels and graph embeddings...")
    labels = np.load("/content/train_y.npy")
    db_embeddings = np.load("/content/Graph_Retriever/embeddings/molpcba_embeddings.npy")
    faiss.normalize_L2(db_embeddings)

    system = HybridRetrievalSystem(embedding_dim=EMBEDDING_DIM)
    system.build_index(db_embeddings)

    query_embeddings = np.random.random((NUM_QUERY, EMBEDDING_DIM)).astype('float32')
    faiss.normalize_L2(query_embeddings)

    for i, query in enumerate(query_embeddings):
        print(f"\nQuery {i+1}")
        final_indices, final_scores = system.search(query.reshape(1, -1))
        print(f"Top {TOP_N_RESULTS} similar graphs:")
        for idx, score in zip(final_indices, final_scores):
            print(f"  Graph Index: {idx:<6} | Score: {score:.4f}")
