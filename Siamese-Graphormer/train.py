import torch
import torch.optim as optim
from torch_geometric.data import DataLoader, Batch # Import Batch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.transforms import ToUndirected
from model import GraphTransformerEncoder, SiameseNetwork
from loss import JaccardContrastiveLoss
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import Dataset

def jaccard_similarity(y1, y2):
    mask = (~torch.isnan(y1)) & (~torch.isnan(y2))
    y1 = y1[mask]
    y2 = y2[mask]
    if y1.numel() == 0:
        return 0.0
    inter = ((y1 == 1) & (y2 == 1)).sum()
    union = ((y1 == 1) | (y2 == 1)).sum()
    return (inter.float() / (union.float() + 1e-8)).item()


class LabeledPairDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, pos_pairs, neg_pairs):
        self.data = data_list
        self.pairs = [(i, j, 1) for (i, j) in pos_pairs] + [(i, j, 0) for (i, j) in neg_pairs]
        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, label = self.pairs[idx]
        return self.data[i], self.data[j], torch.tensor(label, dtype=torch.float)

def extract_embeddings(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Embeddings"):
            batch = batch.to(device)
            out = model(batch) 
            all_embeddings.append(out.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0) 
    return all_embeddings

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = PygGraphPropPredDataset(name="ogbg-molpcba", transform=ToUndirected())
    split_idx = dataset.get_idx_split()
    train_graphs = dataset[split_idx["train"]]
    valid_graphs = [g for g in train_graphs if torch.any((g.y == 1) & ~torch.isnan(g.y))]
    print(f"Filtered {len(valid_graphs)} valid graphs.")

    pos_pairs, neg_pairs = [], []
    for i in range(1000): 
        for j in range(i + 1, 1000):
            sim = jaccard_similarity(valid_graphs[i].y, valid_graphs[j].y)
            if sim >= 0.3:
                pos_pairs.append((i, j))
            elif sim <= 0.05:
                neg_pairs.append((i, j))
    print(f"{len(pos_pairs)} positive and {len(neg_pairs)} negative pairs.")

    pair_dataset = LabeledPairDataset(valid_graphs, pos_pairs, neg_pairs)
    print("Dataset Ready")
    train_loader = DataLoader(pair_dataset, batch_size=64, shuffle=True, num_workers=4)

    node_feature_dim = dataset.num_node_features
    graph_embedding_dim = 256

    encoder = GraphTransformerEncoder(
        in_channels=node_feature_dim,
        hidden_channels=256,
        out_channels=graph_embedding_dim,
        num_layers=3,
        num_heads=8
    ).to(device)

    siamese_net = SiameseNetwork(encoder).to(device)
    criterion = JaccardContrastiveLoss()
    optimizer = optim.Adam(siamese_net.parameters(), lr=1e-4)

    print("Starting training...")
    for epoch in range(1, 16):
        total_loss = 0
        siamese_net.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for data1, data2, sim_label in pbar:
            data1 = data1.to(device)
            data2 = data2.to(device)
            label1 = data1.y.to(device)
            label2 = data2.y.to(device)
            sim_label = sim_label.to(device)  

            optimizer.zero_grad()
            output1, output2 = siamese_net(data1, data2)
            loss = criterion(output1, output2, label1, label2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(siamese_net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")

    print("Training finished.")
    torch.save(siamese_net.state_dict(), "siamese_model.pt")
    torch.save(encoder.state_dict(), "encoder_only.pt")
    encoder.eval() 
    full_dataset = dataset[split_idx["train"]]
    test_dataset =  dataset[split_idx["test"]]
    embeddings = extract_embeddings(encoder, full_dataset, device)
    embeddings_test = extract_embeddings(encoder, test_dataset, device) 
    torch.save(embeddings, "molpcba_embeddings.pt")
    torch.save(embeddings_test, "molpcba_embeddings_test.pt")
    np.save("molpcba_embeddings.npy", embeddings.numpy())
    print("Saved embeddings for FAISS retrieval.")

if __name__ == '__main__':
    train()