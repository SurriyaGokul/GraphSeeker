from atom_encoder import atom_encoder
import random
import copy
from augment import drop_edges, feature_mask
from torch_geometric.data import Batch
import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def balanced_siamese_collate(batch, delta=0.02):
    g1_list, g2_list, labels = [], [], []

    for g in batch:
        original_x = g.x.squeeze().long().to(device)
        original_edge_attr = g.edge_attr.squeeze().long().to(device)
        original_edge_index = g.edge_index.to(device)
        if hasattr(g, 'batch') and g.batch is not None:
            original_batch = g.batch.to(device)
        else:
            original_batch = torch.zeros(g.num_nodes, dtype=torch.long).to(device)

        # First augmentation
        g1 = Batch(
            x=atom_encoder(original_x).detach(),
            edge_attr=F.one_hot(original_edge_attr, num_classes=5).float().detach(),
            edge_index=original_edge_index.clone(),
            batch=original_batch
        )
        g1.edge_index, g1.edge_attr = drop_edges(g1.edge_index, g1.edge_attr, p=0.2)
        g1.x = feature_mask(g1.x, p=0.1)

        # Second augmentation
        g2 = Batch(
            x=atom_encoder(original_x).detach(),
            edge_attr=F.one_hot(original_edge_attr, num_classes=5).float().detach(),
            edge_index=original_edge_index.clone(),
            batch=original_batch
        )
        g2.edge_index, g2.edge_attr = drop_edges(g2.edge_index, g2.edge_attr, p=0.2)
        g2.x = feature_mask(g2.x, p=0.1)

        g1_list.append(g1)
        g2_list.append(g2)
        labels.append(torch.tensor(1.0, device=device))  # Now directly on GPU

    return Batch.from_data_list(g1_list), Batch.from_data_list(g2_list), torch.stack(labels)

