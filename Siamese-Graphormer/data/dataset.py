import torch
from torch.utils.data import Dataset
from collections import defaultdict
import random

class OnlineSiameseSampler(Dataset):
    def __init__(self, dataset, num_pairs=100_000):
        self.dataset   = dataset
        self.num_pairs = num_pairs

        # 1) build index lists over the real dataset
        self.class_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            y = dataset[idx].y
            if not torch.isnan(y):
                self.class_to_indices[int(y)].append(idx)

        # 2) compute “inverse‐freq” weights for each class
        classes = list(self.class_to_indices.keys())
        counts  = torch.tensor([len(self.class_to_indices[c]) for c in classes], 
                               dtype=torch.float)
        inv_freq = 1.0 / counts
        # normalize to sum to 1
        self.classes       = classes
        self.class_weights = (inv_freq / inv_freq.sum()).tolist()

        # cache for speed
        self.available_classes = set(classes)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        # 50/50 pos/neg
        is_pos = (idx % 2 == 0)

        if is_pos:
            # pick a single class *with prob ∝ 1/count*
            cls = random.choices(self.classes, weights=self.class_weights, k=1)[0]
            i, j = random.sample(self.class_to_indices[cls], 2)
            label = 1.0

        else:
            # pick two *different* classes, each ∝ 1/count
            cls1 = random.choices(self.classes, weights=self.class_weights, k=1)[0]
            cls2 = random.choices(self.classes, weights=self.class_weights, k=1)[0]
            # if by bad luck we got the same twice, re-roll until distinct:
            while cls2 == cls1:
                cls2 = random.choices(self.classes, weights=self.class_weights, k=1)[0]

            i = random.choice(self.class_to_indices[cls1])
            j = random.choice(self.class_to_indices[cls2])
            label = 0.0

        g1 = self.dataset[i]
        g2 = self.dataset[j]
        return g1, g2, torch.tensor(label, dtype=torch.float)
