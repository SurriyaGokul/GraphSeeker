import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader, Data
import numpy as np
import random
import pickle
import faiss
import yaml
from tqdm import tqdm
from utils.atom_encoder import SimpleAtomEncoder

from network.re_ranker import CrossEncoderGNN
from network.hybrid_retrieval import HybridRetrievalSystem
from utils.graph_utils import preprocess_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
atom_encoder = SimpleAtomEncoder(emb_dim = 64).to(device)

# Joint graph builder
def build_joint_graph(q_data, c_data, label_diff=None):
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

    cross_attr = torch.zeros((cross_edges.size(1), edge_attr.size(1))) # Setting cross edges to zero
    edge_attr = torch.cat([edge_attr, cross_attr], dim=0)

    batch = torch.zeros(N_q + N_c, dtype=torch.long)
    y = torch.tensor([label_diff], dtype=torch.float) if label_diff is not None else None
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch, y=y)

# Dataset for label difference regression
def build_label_diff_dataset(graphs, labels, num_pairs=20000):
    mean, std = np.mean(labels), np.std(labels)
    norm_labels = (labels - mean) / std
    print("Sample y values (normalized):", norm_labels[:10])

    pairs = []
    for _ in tqdm(range(num_pairs), desc="Building pairs"):
        i, j = random.sample(range(len(graphs)), 2)
        label_diff = norm_labels[i] - norm_labels[j]
        joint_graph = build_joint_graph(graphs[i], graphs[j], label_diff)
        pairs.append(joint_graph)
    return pairs

def train_re_ranker(model, loader, optimizer, criterion, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(pred, data.y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    with open("Graph_Retriever/config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    epochs = config["RE_RANKER_EPOCHS"]
    embedding_dim = config["EMBEDDING_DIM"]
    batch_size = config["RE_RANKER_BATCH_SIZE"]
    node_dim = 64 # This should match your atom encoder output dimension

    graphs = list(ZINC(root='data/ZINC', subset=False, split='train'))
    with open("embeddings/train_graph_embeddings.pkl", "rb") as f:
        embedding_data = pickle.load(f)

    embeddings = np.stack([v["embedding"] for v in embedding_data.values()]).astype("float32")
    labels = np.array([v["label"] for v in embedding_data.values()])
    assert len(graphs) == len(labels) # simple sanity check

    dataset = build_label_diff_dataset(graphs, labels, num_pairs=2000)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CrossEncoderGNN(node_dim=node_dim, edge_dim=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.MSELoss()

    print(f"\nTraining re-ranker model")
    train_re_ranker(model, loader, optimizer, criterion, device, epochs)

    torch.save(model.state_dict(), "checkpoints/reranker_labeldiff_regression.pth")
    print("Model saved: reranker_labeldiff_regression.pth")
