import os
import random
import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.datasets import ZINC
from torch_geometric.data import Batch
from torch_geometric.nn.models import GIN

from network.siamese import SiameseGraphNetwork
from loss.loss import improved_contrastive_loss, nt_xent_loss
from utils.atom_encoder import SimpleAtomEncoder
from utils.augment import drop_edges, feature_mask
from utils.collate import balanced_siamese_collate
from utils.visualization import visualize_embeddings, plot_metrics

device = torch.device("cuda")
atom_encoder = SimpleAtomEncoder(emb_dim=64).to(device)


def train(epochs=10, lr=2e-3, batch_size=256, embeddings_dim=512):
    random.seed(42)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)

    siamese_net = SiameseGraphNetwork(
        in_channels=64,
        edge_dim=5,
        hidden_channels=256,
        out_channels=embeddings_dim
    ).to(device)

    optimizer = optim.Adam(siamese_net.parameters(), lr=lr)

    train_dataset = ZINC(root='data/ZINC', subset=False, split='train')
    print(f"Training on {len(train_dataset)} samples")

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=balanced_siamese_collate
    )

    losses, pos_sims_epoch, neg_sims_epoch, aucs = [], [], [], []

    for epoch in range(1, epochs + 1):
        siamese_net.train()
        total_loss = 0.0
        all_labels, all_sims = [], []

        loop = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch1, batch2, label in loop:
            batch1, batch2, label = batch1.to(device), batch2.to(device), label.to(device)

            out1, out2 = siamese_net(batch1, batch2)
            out1 = F.normalize(out1, p=2, dim=1)
            out2 = F.normalize(out2, p=2, dim=1)

            sims = F.cosine_similarity(out1, out2)
            pos_sims = sims[label == 1]
            neg_sims = sims[label == 0]

            loss = nt_xent_loss(out1, out2, temperature=0.1)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(siamese_net.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            all_labels.extend(label.detach().cpu().numpy())
            all_sims.extend(sims.detach().cpu().numpy())

            loop.set_postfix(
                loss=loss.item(),
                pos_sim=pos_sims.mean().item() if len(pos_sims) else 0,
                neg_sim=neg_sims.mean().item() if len(neg_sims) else 0
            )

        avg_loss = total_loss / len(loader)
        avg_pos = np.mean([x for i, x in enumerate(all_sims) if all_labels[i] == 1])
        avg_neg = np.mean([x for i, x in enumerate(all_sims) if all_labels[i] == 0])

        losses.append(avg_loss)
        pos_sims_epoch.append(avg_pos)
        neg_sims_epoch.append(avg_neg)

        print(f"\nEpoch {epoch}:")
        print(f"Avg Positive Similarity = {avg_pos:.4f}")
        print(f"Avg Negative Similarity = {avg_neg:.4f}")
        print(f"Gradient Norm = {grad_norm:.4f}")

    # Save Model
    torch.save(siamese_net.state_dict(), "checkpoints/siamese_final.pt")
    print("\n Model saved at checkpoints/siamese_final.pt")

    # Save Metrics Plot
    plot_metrics(losses, pos_sims_epoch, neg_sims_epoch, aucs)
    print("Training curve saved at checkpoints/training_metrics.png")

    # Save Embeddings
    siamese_net.eval()
    embedding_dict = {}
    with torch.no_grad():
        full_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: Batch.from_data_list(x)
        )
        for i, graph in tqdm(enumerate(full_loader), total=len(train_dataset), desc="Generating Embeddings"):
            graph = graph.to(device)
            graph.x = atom_encoder(graph.x.squeeze().long()).to(device)
            graph.edge_attr = F.one_hot(graph.edge_attr.squeeze().long(), num_classes=5).float().to(device)
            embedding = siamese_net.encoder(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
            embedding_dict[i] = {
            "embedding": embedding.squeeze().cpu().numpy(),
            "label": graph.y.item()  
        }


    with open("embeddings/train_graph_embeddings.pkl", "wb") as f:
        pickle.dump(embedding_dict, f)
    print("Embeddings saved at embeddings/train_graph_embeddings.npy")

    # Visualize t-SNE
    visualize_embeddings(embedding_dict)
    print("t-SNE saved at embeddings/tsne.png")


if __name__ == "__main__":
    train()