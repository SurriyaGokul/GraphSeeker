import torch.nn.functional as F

def improved_contrastive_loss(out1, out2, labels, margin=0.3, scale=5.0):
    sims = F.cosine_similarity(out1, out2) * scale
    pos_loss = (1 - sims / scale) ** 2
    neg_loss = F.relu(sims / scale - margin) ** 2
    loss = labels * pos_loss + (1 - labels) * neg_loss
    return loss.mean()
