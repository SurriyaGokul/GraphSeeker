import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from utils.atom_encoder import SimpleAtomEncoder
import numpy as np
from tqdm import tqdm
from utils.graph_utils import preprocess_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
atom_encoder = SimpleAtomEncoder().to(device)
def preprocess_graph(graph):
    with torch.no_grad():
        graph = graph.clone()
        graph.x = atom_encoder(graph.x.squeeze().long())
        graph.edge_attr = F.one_hot(graph.edge_attr.squeeze().long(), num_classes=5).float()
        return graph
