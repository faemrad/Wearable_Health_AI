from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .paths import REPO_ROOT
from .preprocess import founder_1lead, hubert_input


class HRWindows(Dataset):
    def __init__(self, manifest: pd.DataFrame, kind: str, random_crop: bool = False):
        self.df = manifest.reset_index(drop=True)
        self.kind = kind
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        ecg = np.load(REPO_ROOT / row["npy"])
        y = np.float32(row["hr_bpm"])
        if self.kind == "founder":
            x = founder_1lead(ecg)
        elif self.kind == "hubert":
            x = hubert_input(ecg, random_crop=self.random_crop)
        else:
            raise ValueError(self.kind)
        return torch.from_numpy(x.copy()), torch.tensor([y], dtype=torch.float32)
