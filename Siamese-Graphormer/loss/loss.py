import torch.nn.functional as F

def pairwise_contrastive_loss(out1, out2, label, margin=1.0):
    """
    out1, out2: (B, D)
    label: (B,) — 1 for same class (positive), 0 for different class (negative)
    """
    euclidean_distance = F.pairwise_distance(out1, out2)
    loss = label * euclidean_distance.pow(2) + (1 - label) * F.relu(margin - euclidean_distance).pow(2)
    return loss.mean()
