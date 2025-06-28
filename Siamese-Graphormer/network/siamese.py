import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import GraphTransformerEncoder


class SiameseGraphNetwork(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_channels, out_channels, num_layers=3, num_heads=8):
        super().__init__()
        self.encoder = GraphTransformerEncoder(
            in_channels, edge_dim, hidden_channels, out_channels, num_layers, num_heads
        )

    def forward(self, data1, data2):
        out1 = self.encoder(data1.x.float(), data1.edge_index, data1.edge_attr, data1.batch)
        out2 = self.encoder(data2.x.float(), data2.edge_index, data2.edge_attr, data2.batch)
        return out1, out2
