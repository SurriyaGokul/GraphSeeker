import torch
import torch.nn.functional as F

def improved_contrastive_loss(out1, out2, labels, margin=0.3, scale=5.0):
    sims = F.cosine_similarity(out1, out2) * scale
    pos_loss = (1 - sims / scale) ** 2
    neg_loss = F.relu(sims / scale - margin) ** 2
    loss = labels * pos_loss + (1 - labels) * neg_loss
    return loss.mean()

def nt_xent_loss(z1, z2, temperature=0.1):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    # Normalize to unit vectors
    z = F.normalize(z, p=2, dim=1)

    sim_matrix = torch.matmul(z, z.T) / temperature

    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix.masked_fill_(mask, float('-inf'))

    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)

    loss = F.cross_entropy(sim_matrix, targets)
    return loss
