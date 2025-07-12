import torch
from torch_geometric.nn import GINEConv, global_add_pool
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm

class CrossEncoderGNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=128, num_layers=4):
        super().__init__()

        self.gnns = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        edge_nn = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gnns.append(GINEConv(nn.Linear(node_dim, hidden_dim), edge_dim=edge_dim))
        self.bns.append(BatchNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            edge_nn = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.gnns.append(GINEConv(nn.Linear(hidden_dim, hidden_dim), edge_dim=edge_dim))
            self.bns.append(BatchNorm(hidden_dim))

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = x
        for gnn, bn in zip(self.gnns, self.bns):
            h = gnn(h, edge_index, edge_attr)
            h = bn(h)
            h = torch.relu(h)  

        pooled = global_mean_pool(h, batch)
        return self.regressor(pooled).squeeze(-1)
