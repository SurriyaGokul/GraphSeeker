import torch
from torch_geometric.nn import GINEConv, global_add_pool
import torch.nn as nn

class CrossEncoderGNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128):
        super().__init__()

        self.gnn_layers = torch.nn.ModuleList([
            GINEConv(nn=nn.Linear(node_dim, hidden_dim), edge_dim=edge_dim),
            GINEConv(nn=nn.Linear(hidden_dim, hidden_dim), edge_dim=edge_dim)
        ])

        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = x
        for gnn in self.gnn_layers:
            h = gnn(h, edge_index, edge_attr)
        pooled = global_add_pool(h, batch)
        return self.regressor(pooled).squeeze(-1) 
