import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os

def plot_metrics(losses, pos_sims, neg_sims, aucs):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label="NT-Xent Loss")
    plt.plot(pos_sims, label="Avg Positive Sim")
    plt.plot(neg_sims, label="Avg Negative Sim")
    plt.plot(aucs, label="AUC")
    plt.xlabel("Epoch")
    plt.title("Training Metrics")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("checkpoints/training_metrics.png")
    plt.close()


def visualize_embeddings(embedding_dict):
    embeddings = np.stack(list(embedding_dict.values()))
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.6)
    plt.title("t-SNE Visualization of Graph Embeddings")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("embeddings/tsne.png")
    plt.close()