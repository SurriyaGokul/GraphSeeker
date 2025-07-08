import torch

def feature_mask(x, p=0.1):
    mask = torch.rand_like(x[:, 0]) < p
    x[mask] = 0
    return x

def drop_edges(edge_index, edge_attr=None, p=0.2):
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges) > p
    edge_index = edge_index[:, keep_mask]
    if edge_attr is not None:
        edge_attr = edge_attr[keep_mask]
    return edge_index, edge_attr