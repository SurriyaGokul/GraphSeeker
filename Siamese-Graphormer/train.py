import torch
import torch.optim as optim
from torch_geometric.datasets import ZINC
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from network.siamese import SiameseGraphNetwork
from loss.loss import improved_contrastive_loss,nt_xent_loss
from tqdm import tqdm
import random
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from utils.collate import balanced_siamese_collate
from atom_encoder import atom_encoder   


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(epochs=1, lr=2e-3, batch_size=256, embeddings_dim=512):
    random.seed(42)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)

    siamese_net = SiameseGraphNetwork(
        in_channels=64,      
        edge_dim=5,             # ← one-hot bond type
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
        num_workers=2,
        drop_last=True,
        collate_fn=balanced_siamese_collate
    )

    for epoch in range(1, epochs + 1):
        siamese_net.train()
        total_loss = 0.0
        all_labels = []
        all_sims = []

        loop = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch1, batch2, label in loop:
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            label = label.to(device)

            out1, out2 = siamese_net(batch1, batch2)
            out1 = F.normalize(out1, p=2, dim=1)
            out2 = F.normalize(out2, p=2, dim=1)
            print("Out1 mean:", out1.mean().item(), "std:", out1.std().item())
            print("Out2 mean:", out2.mean().item(), "std:", out2.std().item())

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
                pos_sim=pos_sims.mean().item() if len(pos_sims) > 0 else 0,
                neg_sim=neg_sims.mean().item() if len(neg_sims) > 0 else 0
            )

        avg_loss = total_loss / len(loader)

        print(f"\nEpoch {epoch}: Avg Loss = {avg_loss:.4f}, AUC = {auc_score:.4f}")
        print(f"  Final avg pos sim = {pos_sims.mean():.4f}, avg neg sim = {neg_sims.mean():.4f}")
        print(f"  Grad Norm = {grad_norm:.4f}")

    # Save Model
    torch.save(siamese_net.state_dict(), "checkpoints/siamese_final.pt")
    print("Model weights saved at checkpoints/siamese_final.pt")

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
            graph.x = atom_encoder(graph.x.squeeze().long()) 
            graph.edge_attr = F.one_hot(graph.edge_attr.squeeze().long(), num_classes=5).float()
            embedding = siamese_net.encoder(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
            embedding_dict[i] = embedding.squeeze().cpu().numpy()

    np.save("embeddings/train_graph_embeddings.npy", embedding_dict)
    print("Graph embeddings saved at embeddings/train_graph_embeddings.npy")

if __name__=="__main__":
    train()