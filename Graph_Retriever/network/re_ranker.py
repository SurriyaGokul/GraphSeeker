import torch
from torch_geometric.nn import GINConv, global_add_pool

class CrossEncoderGNN(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim=128):
        super().__init__()
        self.gnn_layers = torch.nn.ModuleList([
            GINConv(torch.nn.Linear(node_dim, hidden_dim)),
            GINConv(torch.nn.Linear(hidden_dim, hidden_dim)),
        ])
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, joint_x, joint_edge_index, joint_batch):
        h = joint_x
        for gnn in self.gnn_layers:
            h = gnn(h, joint_edge_index)
        pooled = global_add_pool(h, joint_batch)
        score = self.classifier(pooled)
        return torch.sigmoid(score).squeeze(-1)
