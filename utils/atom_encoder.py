import torch.nn as nn

class SimpleAtomEncoder(nn.Module):
    def __init__(self, emb_dim=32, num_embeddings=119):  # 119 = max atomic number + 1
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, emb_dim)

    def forward(self, x):
        return self.emb(x.squeeze(-1).long()) 
