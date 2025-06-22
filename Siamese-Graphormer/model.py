import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = out_channels // num_heads

        self.W_q = nn.Linear(in_channels, out_channels)
        self.W_k = nn.Linear(in_channels, out_channels)
        self.W_v = nn.Linear(in_channels, out_channels)
        self.W_o = nn.Linear(out_channels, out_channels)

        self.input_proj = nn.Linear(in_channels, out_channels)
        self.layer_norm1 = nn.LayerNorm(out_channels)
        self.layer_norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(p=0.1)

        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.GELU(),
            nn.Linear(out_channels * 2, out_channels),
        )

    def forward(self, x, edge_list):
        q = self.W_q(x).view(-1, self.num_heads, self.attention_dim)
        k = self.W_k(x).view(-1, self.num_heads, self.attention_dim)
        v = self.W_v(x).view(-1, self.num_heads, self.attention_dim)

        alpha = torch.einsum('ihd,jhd->hij', q, k) / (self.attention_dim ** 0.5)

        adj = torch.zeros(x.size(0), x.size(0), device=x.device)
        adj[edge_list[0], edge_list[1]] = 1
        adj[torch.arange(x.size(0)), torch.arange(x.size(0))] = 1

        mask = adj.unsqueeze(0) == 0
        alpha = alpha.masked_fill(mask, float('-inf'))
        alpha = F.softmax(alpha, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)

        attn_output = torch.einsum("hij,jhd->ihd", alpha, v).reshape(x.size(0), -1)

        x = self.input_proj(x)
        x = self.layer_norm1(x + self.W_o(attn_output))
        x = self.layer_norm2(x + self.dropout(self.ffn(x)))

        return x

class GraphTransformerEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphTransformerLayer(in_channels, hidden_channels, num_heads))

        for _ in range(num_layers - 2):
            self.layers.append(GraphTransformerLayer(hidden_channels, hidden_channels, num_heads))

        self.layers.append(GraphTransformerLayer(hidden_channels, out_channels, num_heads))

    def forward(self, x, edge_list, batch):
        for layer in self.layers:
            x = layer(x, edge_list)
        return global_mean_pool(x, batch)

class SiameseNetwork(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, data1, data2):
        out1 = self.encoder(data1.x.float(), data1.edge_index, data1.batch)
        out2 = self.encoder(data2.x.float(), data2.edge_index, data2.batch)
        return out1, out2
