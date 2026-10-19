import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class Datasets(Dataset):
    def __init__(self, signals, labels, train=False, noise_std=0.01,
                 scale_range=(0.9, 1.1), max_shift_frac=0.05, mask_frac=0.05):
        self.signals = signals
        self.labels = labels
        self.train = train
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.max_shift_frac = max_shift_frac
        self.mask_frac = mask_frac

    def __len__(self):
        return len(self.signals)

    def _augment(self, x):
        seq_len = x.shape[-1]

        scale = np.random.uniform(*self.scale_range)
        x = x * scale

        if self.noise_std > 0:
            x = x + np.random.normal(0, self.noise_std, size=x.shape).astype(x.dtype)

        if self.max_shift_frac > 0:
            max_shift = int(seq_len * self.max_shift_frac)
            if max_shift > 0:
                shift = np.random.randint(-max_shift, max_shift + 1)
                x = np.roll(x, shift, axis=-1)

        if self.mask_frac > 0:
            mask_len = int(seq_len * self.mask_frac)
            if mask_len > 0:
                start = np.random.randint(0, seq_len - mask_len + 1)
                x = x.copy()
                x[..., start:start + mask_len] = 0.0

        return x

    def __getitem__(self, idx):
        x = self.signals[idx]
        if x.ndim == 1:
            x = x[np.newaxis, :]

        x = np.asarray(x, dtype=np.float32)

        if self.train:
            x = self._augment(x)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y