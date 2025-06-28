import faiss
import numpy as np  
import torch
from network.hybrid_retrieval import HybridRetrievalSystem
from config import EMBEDDING_DIM
from typing import List, Tuple

def get_similar(query_embedding: np.ndarray, k: int = 5):
    # Load stored embeddings and labels
    data = torch.load(r"C:\Users\Surrya Gokul\OneDrive\Desktop\Scalable_Graph_Retrieval\Scalable_Graph_Retrieval\Graph_Retriever\embeddings\graph_embeddings.pt")
    embeddings = data['embeddings'].numpy().astype('float32')
    labels = data['labels'].numpy().flatten()

    # Balance classes by sampling equal number of positive/negative examples
    positive_embeddings = embeddings[labels == 1]
    negative_embeddings = embeddings[labels == 0]
    min_k = min(len(positive_embeddings), len(negative_embeddings))

    pos_sample = positive_embeddings[np.random.choice(len(positive_embeddings), min_k, replace=False)]
    neg_sample = negative_embeddings[np.random.choice(len(negative_embeddings), min_k, replace=False)]

    balanced_embeddings = np.concatenate([pos_sample, neg_sample], axis=0)

    # Normalize embeddings before FAISS indexing
    faiss.normalize_L2(balanced_embeddings)

    # Build index and search
    system = HybridRetrievalSystem(embedding_dim=EMBEDDING_DIM, training=False)
    system.build_index(balanced_embeddings)

    # Normalize query
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    indices, distances = system.search(query_embedding)
    
    print(f"Top {k} similar graphs:")
    for idx, dist in zip(indices[:k], distances[:k]):
        print(f"Graph Index: {idx:<6} | Score: {dist:.4f}")

    return indices[:k], distances[:k]


if __name__ == "__main__":
    # Example query embedding
    query_embedding = np.random.random((1, EMBEDDING_DIM)).astype('float32')
    get_similar(query_embedding, k=10)
