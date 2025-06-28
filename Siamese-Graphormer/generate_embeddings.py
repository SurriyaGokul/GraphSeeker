import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset
from model import GraphTransformerEncoder  
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_NAME = "ogbg-molpcba"
BATCH_SIZE = 64
EMBEDDING_DIM = 256

dataset = PygGraphPropPredDataset(name=DATASET_NAME, root='data/')
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

encoder = GraphTransformerEncoder(
    in_channels=dataset.num_node_features,
    hidden_channels=EMBEDDING_DIM,
    out_channels=EMBEDDING_DIM
).to(DEVICE)

encoder.load_state_dict(torch.load("models/graphormer_encoder.pth"))
encoder.eval()

all_embeddings = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Generating Embeddings"):
        batch = batch.to(DEVICE)
        emb = encoder(batch.x.float(), batch.edge_index, batch.batch)
        all_embeddings.append(emb.cpu())

embeddings = torch.cat(all_embeddings, dim=0).numpy()

os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/molpcba_embeddings.npy", embeddings)

print(f"Saved {embeddings.shape[0]} embeddings to 'embeddings/molpcba_embeddings.npy'")
