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
