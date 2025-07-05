import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

class EdgeConditionedGraphAttention(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        self.W_q = nn.Linear(in_channels, out_channels)
        self.W_k = nn.Linear(in_channels, out_channels)
        self.W_v = nn.Linear(in_channels, out_channels)
        self.edge_proj = nn.Linear(edge_dim, out_channels)
        self.out_proj = nn.Linear(out_channels, out_channels)

        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.2)

        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.GELU(),
            nn.Linear(out_channels * 2, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index,
                                               edge_attr=edge_attr,
                                               fill_value=0.0)

        num_nodes = x.size(0)

        q = self.W_q(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.W_k(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.W_v(x).view(num_nodes, self.num_heads, self.head_dim)

        src, dst = edge_index
        q_dst = q[dst]   # [E, H, D]
        k_src = k[src]
        v_src = v[src]

        edge_e = self.edge_proj(edge_attr).view(-1, self.num_heads, self.head_dim)

        scores = (q_dst * (k_src + edge_e)).sum(dim=-1) / (self.head_dim ** 0.5)
        alpha  = F.softmax(scores, dim=0)            # over all edges

        messages = alpha.unsqueeze(-1) * (v_src + edge_e)  # [E, H, D]

        out_sum = x.new_zeros((num_nodes, self.num_heads, self.head_dim))
        out_sum.index_add_(0, dst, messages)               

        counts = torch.bincount(dst, minlength=num_nodes).clamp(min=1)
        counts = counts.view(-1, 1, 1).to(out_sum.dtype)

        out = out_sum / counts                               # [N, H, D]
        out = out.view(num_nodes, -1)                        # [N, H*D]

        x = self.norm1(x + self.out_proj(out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x
