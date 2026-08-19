import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ECGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx]
        if x.ndim == 1:
            x = x[np.newaxis, :]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long) 
        return x, y