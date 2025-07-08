from atom_encoder import atom_encoder
import random
import copy
from augment import drop_edges, feature_mask
from torch_geometric.data import Batch
import torch
import torch.nn.functional as F

def balanced_siamese_collate(batch, delta=0.02):
    pairs = []
    num_samples = len(batch)

    for g in batch:
        g.x = atom_encoder(g.x.squeeze().long()).detach()
        g.edge_attr = F.one_hot(g.edge_attr.squeeze().long(), num_classes=5).float().detach()

    for _ in range(num_samples):
        g1 = random.choice(batch)
        g2 = copy.deepcopy(g1)

        # Apply augmentations
        for g in [g1, g2]:
            g.edge_index, g.edge_attr = drop_edges(g.edge_index, g.edge_attr, p=0.2)
            g.x = feature_mask(g.x, p=0.1)

        pairs.append((g1, g2, torch.tensor(1.0)))  # Always positive in NT-Xent

    g1_list, g2_list, labels = zip(*pairs)
    batch1 = Batch.from_data_list(g1_list)
    batch2 = Batch.from_data_list(g2_list)
    labels = torch.stack(labels)
    return batch1, batch2, labels