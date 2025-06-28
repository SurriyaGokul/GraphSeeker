import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred import Evaluator 
from data.dataset import OnlineSiameseSampler
from network.siamese import SiameseGraphNetwork
from loss.loss import pairwise_contrastive_loss
from tqdm import tqdm
import random
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def siamese_collate(batch):
    g1_list, g2_list, labels = zip(*batch)
    batch1 = Batch.from_data_list(g1_list)
    batch2 = Batch.from_data_list(g2_list)
    labels = torch.stack(labels)        # shape [B]
    return batch1, batch2, labels

def train(epochs = 10,lr= 1e-3,batch_size = 32,embeddings_dim = 256):
    # Set random seeds for reproducibility
    random.seed(42)
    siamese_net = SiameseGraphNetwork(in_channels = 9, edge_dim = 3, hidden_channels = 256, embeddings_dim).to(device)
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=lr)

    # Load the OGB dataset
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='data/ogb_molhiv')
    # Split the dataset into training and validation sets
    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx['train']]
    print(f"Training on {len(train_dataset)} samples")

    # wrap sampler in a DataLoader once
    loader = DataLoader(
        OnlineSiameseSampler(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=siamese_collate,
        drop_last=True
    )

    for epoch in range(1, epochs+1):
        siamese_net.train()
        total_loss = 0.0

        loop = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch1, batch2, label in loop:
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            label  = label.to(device)

            out1, out2 = siamese_net(batch1, batch2)
            loss = pairwise_contrastive_loss(out1, out2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")


if __name__ == '__main__':
    train()