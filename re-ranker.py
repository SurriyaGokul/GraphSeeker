import torch
import torch.nn as nn
class ReRankerNet(nn.Module):
    def __init__(self, embedding_dim):
        super(ReRankerNet, self).__init__()
        self.layer1 = nn.Linear(embedding_dim * 4, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 1)
        self.activation = nn.ReLU()

    def forward(self, q, c):
        diff = torch.abs(q - c)
        prod = q * c
        combined = torch.cat([q, c, diff, prod], dim=1)
        
        x = self.activation(self.layer1(combined))
        x = self.activation(self.layer2(x))
        x = torch.sigmoid(self.layer3(x)) 
        return x