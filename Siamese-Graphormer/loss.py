import torch
import torch.nn as nn
import torch.nn.functional as F

class JaccardContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0, threshold: float = 0.3):
        super().__init__()
        self.margin = margin
        self.threshold = threshold  

    def forward(self, emb1, emb2, label1, label2):
        assert emb1.shape == emb2.shape
        assert label1.shape == label2.shape

        device = emb1.device

        # Masking out the NaN values in labels
        valid_mask = (~torch.isnan(label1)) & (~torch.isnan(label2))
        label1_clean = torch.where(valid_mask, label1, torch.tensor(0.0, device=device))
        label2_clean = torch.where(valid_mask, label2, torch.tensor(0.0, device=device))

        label1_bin = (label1_clean > 0.5).float()
        label2_bin = (label2_clean > 0.5).float()

        # Jaccard similarity
        intersection = (label1_bin * label2_bin).sum(dim=1)
        union = ((label1_bin + label2_bin) >= 1).float().sum(dim=1) + 1e-8
        jaccard_sim = intersection / union  
        is_similar = (jaccard_sim >= self.threshold).float()  
        distance = F.pairwise_distance(emb1, emb2)
        loss_similar = is_similar * distance.pow(2)
        loss_dissimilar = (1 - is_similar) * F.relu(self.margin - distance).pow(2)

        return (loss_similar + loss_dissimilar).mean()