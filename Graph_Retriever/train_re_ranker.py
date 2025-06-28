import torch
import yaml
from torch_geometric.data import Data, DataLoader
from network.re_ranker import CrossEncoderGNN
from network.hybrid_retrieval import HybridRetrievalSystem
import numpy as np
import faiss
import random
from typing import List

def build_joint_graph(q_data, c_data, label, add_cross_edges=True):
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
    return Data(x=joint_x, edge_index=joint_edge_index, batch=joint_batch, y=torch.tensor([label], dtype=torch.float))

def build_balanced_joint_dataset(graphs: List[Data], embeddings, labels, num_samples=1000):
    system = HybridRetrievalSystem(embedding_dim=embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    system.build_index(embeddings)

    joint_graphs = []
    pos_count = 0
    neg_count = 0
    max_each = num_samples // 2

    for i, query_graph in enumerate(graphs):
        query_embedding = embeddings[i].reshape(1, -1)
        retrieved_indices, _ = system.search(query_embedding)

        for idx in retrieved_indices:
            if idx == i or idx >= len(graphs):
                continue

            label = 1 if labels[i] == labels[idx] else 0

            # Balance condition
            if label == 1 and pos_count >= max_each:
                continue
            if label == 0 and neg_count >= max_each:
                continue

            candidate_graph = graphs[idx]
            joint_graph = build_joint_graph(query_graph, candidate_graph, label)
            joint_graphs.append(joint_graph)

            if label == 1:
                pos_count += 1
            else:
                neg_count += 1

            if pos_count >= max_each and neg_count >= max_each:
                break

        if len(joint_graphs) >= num_samples:
            break

    random.shuffle(joint_graphs)
    return joint_graphs

def train_re_ranker(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data.x, data.edge_index, data.batch)
        loss = criterion(outputs, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def load_graph_dataset():
    # This function should load your actual graph data objects
    return torch.load("embeddings/graph_data.pt")  # Assuming graph objects are stored separately here

if __name__ == "__main__":

    with open("Graph_Retriever/config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    epochs = config['RE_RANKER_EPOCHS']
    batch_size = config['RE_RANKER_BATCH_SIZE']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load embeddings and the labels
    data = torch.load("embeddings/graph_embeddings.pt")
    embeddings = data['embeddings'].numpy().astype('float32')
    labels = data['labels'].numpy().flatten()

    # Load actual graph data objects
    graphs = load_graph_dataset()
    assert len(graphs) == len(labels), "Mismatch between number of graphs and labels"

    # Building balanced train dataset
    joint_graphs = build_balanced_joint_dataset(graphs, embeddings, labels, num_samples=2000)
    train_loader = DataLoader(joint_graphs, batch_size=batch_size, shuffle=True)

    # Train re-ranker
    model = CrossEncoderGNN(node_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss() # I am using it because the labels are binary (0 or 1), might need to change if you have different labels

    for epoch in range(epochs):
        loss = train_re_ranker(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), "reranker.pth") # Save the trained model for using in retrieval
    print("Re-ranker training complete and model saved.")
